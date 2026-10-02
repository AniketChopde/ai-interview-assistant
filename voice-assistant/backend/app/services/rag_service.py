import os
from typing import List, Dict, Any, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

load_dotenv()

def _map_jd_name_to_question_file(jd_name: str) -> str:
    """
    Map JD names to their corresponding question file names.
    Handles variations like "Generative AI Developer" -> "genai_developer.txt"
    """
    if not jd_name:
        return ""
    
    # Normalize the JD name
    normalized = jd_name.lower().strip()
    
    # Remove common prefixes/suffixes
    normalized = normalized.replace("job description:", "").strip()
    
    # Map common variations to actual question file names
    # Check for specific patterns first
    if "genai" in normalized or "generative ai" in normalized or "generativeai" in normalized:
        if "senior" in normalized:
            return "genai_developer_senior.txt"
        elif "fresher" in normalized or "junior" in normalized or "entry" in normalized:
            return "genai_developer_fresher.txt"
        elif "engineer" in normalized:
            if "fresher" in normalized or "junior" in normalized:
                return "genai_engineer_fresher.txt"
            return "genai_developer.txt"  # Default for engineer
        else:
            return "genai_developer.txt"
    elif "full stack" in normalized or "fullstack" in normalized:
        if "senior" in normalized:
            return "full_stack_developer_senior.txt"
        elif "fresher" in normalized or "junior" in normalized:
            return "full_stack_developer_fresher.txt"
        else:
            return "full_stack_developer.txt"
    elif "backend" in normalized:
        if "senior" in normalized:
            return "backend_developer_senior.txt"
        elif "fresher" in normalized or "junior" in normalized:
            return "backend_developer_fresher.txt"
        else:
            return "backend_developer.txt"
    elif "frontend" in normalized:
        if "senior" in normalized:
            return "frontend_developer_senior.txt"
        elif "fresher" in normalized or "junior" in normalized:
            return "frontend_developer_fresher.txt"
        else:
            return "frontend_developer.txt"
    
    # Fallback: try to construct from normalized name
    fallback = normalized.replace(' ', '_').replace('-', '_')
    # Remove common words that might not be in filename
    fallback = fallback.replace('_developer', '').replace('_engineer', '')
    fallback = fallback.replace('_position', '').replace('_role', '')
    
    # If it still looks reasonable, use it
    if fallback and len(fallback) > 3:
        return f"{fallback}.txt"
    
    # Last resort: return empty string
    return ""

class RAGService:
    def __init__(self):
        # Set up persistent directory
        curr_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.persistent_dir = os.path.join(curr_dir, "db", "vector_store")
        self.jd_collection_name = "job_descriptions"
        self.resume_collection_name = "resumes"
        
        # Ensure directory exists
        os.makedirs(self.persistent_dir, exist_ok=True)
        
        # Cache for JD texts and embeddings to avoid re-embedding on every match request
        self._jd_files_cache = []
        self._jd_embs_cache = []
        
        # Check for Gemini or OpenAI API keys
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        # Debug logging
        print("[INFO] Environment variables check:")
        print(f"  GEMINI_API_KEY / GOOGLE_API_KEY: {'[SET]' if gemini_key else '[MISSING]'}")
        print(f"  OPENAI_API_KEY: {'[SET]' if openai_key else '[MISSING]'}")
        
        self.openai_configured = bool(gemini_key or openai_key)
        self.ai_configured = self.openai_configured
        
        if not self.openai_configured:
            print("[WARN] Neither Gemini nor OpenAI API key configured. RAG features will be disabled.")
            print("Please set one of the following environment variables:")
            print("- GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini Free Tier")
            print("- OPENAI_API_KEY for OpenAI")
            self.model = None
            self.embeddings = None
            self.jd_vector_store = None
            self.resume_vector_store = None
            self.text_splitter = None
            self.matching_prompt = None
            return
        
        # Initialize LLM and Embeddings
        try:
            if gemini_key and ChatGoogleGenerativeAI and GoogleGenerativeAIEmbeddings:
                print("[INFO] Initializing RAG Service with Google Gemini")
                self.model = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite",
                    google_api_key=gemini_key,
                    temperature=0.7
                )
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=gemini_key
                )
            elif openai_key and ChatOpenAI and OpenAIEmbeddings:
                print("✨ Initializing RAG Service with OpenAI")
                self.model = ChatOpenAI(
                    model="gpt-4o",
                    api_key=openai_key,
                    temperature=0.7
                )
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=openai_key
                )
            else:
                raise ValueError("No valid AI provider client available")
            
            # Initialize vector stores
            self.jd_vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persistent_dir,
                collection_name=self.jd_collection_name
            )
            
            self.resume_vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persistent_dir,
                collection_name=self.resume_collection_name
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Prompt template for matching
            self.matching_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert HR professional and technical recruiter specializing in tech hiring.
                
                TASK: Analyze the match between a candidate's resume and a job description.
                
                RESUME: {resume_content}
                
                JOB DESCRIPTION: {jd_content}
                
                ANALYSIS CRITERIA:
                1. Technical Skills Match (40%): Core languages, frameworks, tools, cloud platforms
                2. Experience Level Match (25%): Years of experience, role seniority, domain relevance
                3. Project/Domain Relevance (20%): Similar projects, industry domain, scale/complexity
                4. Education & Certifications (15%): Relevant degree, certifications, training
                
                OUTPUT FORMAT: Return a JSON object with EXACTLY this structure:
                {{
                    "match_score": <integer between 0 and 100>,
                    "analysis": "<detailed analysis explaining the match score>",
                    "key_matches": ["<specific skill 1>", "<specific skill 2>", ...],
                    "missing_skills": ["<critical missing skill 1>", "<critical missing skill 2>", ...],
                    "recommendations": "<specific recommendations for improvement>"
                }}
                
                IMPORTANT: Only return valid JSON. No additional text."""),
                ("user", "Analyze the match between this resume and job description. Return only JSON.")
            ])
            
            # Auto-initialize job descriptions (cached in vector store if already present)
            try:
                self.load_job_descriptions()
            except Exception as jd_err:
                print(f"[WARN] Could not auto-load job descriptions on startup: {jd_err}")
                
            print("[INFO] RAG system initialized successfully!")
            
        except Exception as e:
            print(f"[ERROR] Error initializing RAG system: {str(e)}")
            self.openai_configured = False
            self.model = None
            self.embeddings = None
            self.jd_vector_store = None
            self.resume_vector_store = None
            self.text_splitter = None
            self.matching_prompt = None
    
    def load_job_descriptions(self, force_reload: bool = False):
        """Load all job descriptions from the jd folder into the vector store"""
        if not self.openai_configured:
            print("[WARN] Cannot load job descriptions: AI provider not configured")
            return
            
        jd_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "jd")
        
        if not os.path.exists(jd_dir):
            print("[WARN] JD directory not found")
            return
            
        # Check if JDs are already indexed in ChromaDB
        try:
            existing_count = self.jd_vector_store._collection.count()
            if existing_count > 0 and not force_reload:
                print(f"[INFO] Vector store already contains {existing_count} JD documents. Skipping re-embedding.")
                return
        except Exception:
            pass
        
        # Clear existing JD documents if force reloading
        if force_reload:
            try:
                self.jd_vector_store._collection.delete(where={"type": "job_description"})
            except Exception as e:
                print(f"[WARN] Could not clear existing JDs: {e}")
        
        jd_documents = []
        for filename in os.listdir(jd_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(jd_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "type": "job_description",
                        "position": filename.replace('.txt', '').replace('_', ' ').title()
                    }
                )
                jd_documents.append(doc)
        
        if jd_documents:
            try:
                splits = self.text_splitter.split_documents(jd_documents)
                self.jd_vector_store.add_documents(splits)
                print(f"[INFO] Loaded {len(jd_documents)} job descriptions into vector store")
            except Exception as e:
                print(f"[WARN] Failed to embed job descriptions: {e}")
    
    def process_resume(self, file_path: str, user_id: str) -> str:
        """Process uploaded resume and store in vector store"""
        if not self.openai_configured:
            # Fallback: just store the file path for later processing
            print("⚠️  OpenAI not configured. Storing resume file for later processing.")
            return "Resume uploaded successfully (RAG processing disabled - OpenAI not configured)"
            
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Split documents
            splits = self.text_splitter.split_documents(docs)
            
            # Add metadata
            for split in splits:
                split.metadata.update({
                    "user_id": user_id,
                    "type": "resume",
                    "source": file_path
                })
            
            # Clear existing resume for this user
            try:
                self.resume_vector_store._collection.delete(where={"user_id": user_id})
            except Exception as e:
                print(f"Warning: Could not clear existing resume for user {user_id}: {e}")
            
            # Add to vector store
            self.resume_vector_store.add_documents(splits)
            
            return "Resume processed successfully"
            
        except Exception as e:
            print(f"Error processing resume: {e}")
            return f"Error processing resume: {str(e)}"
    
    def get_resume_content(self, user_id: str) -> str:
        """Retrieve resume content for a user by user_id (metadata filter).
        Uses Chroma's where clause so we always get this user's resume regardless
        of other documents in the collection (fixes production bug where semantic
        search top-k didn't include the user's docs)."""
        if not self.openai_configured:
            return ""

        try:
            results = self.resume_vector_store._collection.get(
                where={"user_id": user_id},
                include=["documents"],
            )
            if not results or not results.get("documents"):
                return ""

            docs = results["documents"]
            # Chroma may return list of str (one per id) or list of list; flatten if needed
            if docs and isinstance(docs[0], list):
                docs = [d for sublist in docs for d in sublist]
            content = "\n\n".join(docs)
            return content if content else ""

        except Exception as e:
            print(f"Error retrieving resume: {e}")
            return ""
    
    def _get_jd_embeddings_and_files(self) -> tuple:
        """Get or compute and cache all JD texts and their embeddings."""
        if self._jd_files_cache and self._jd_embs_cache:
            return self._jd_files_cache, self._jd_embs_cache
            
        jd_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "jd")
        jd_files = []
        if os.path.exists(jd_dir):
            for filename in os.listdir(jd_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(jd_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        jd_content = f.read()
                    jd_files.append((filename.replace('.txt', '').replace('_', ' ').title(), jd_content))
        
        if not jd_files:
            return [], []
            
        print(f"[INFO] Embedding {len(jd_files)} job descriptions for similarity index...")
        vectors = self.embeddings.embed_documents([jd_text for _, jd_text in jd_files])
        jd_embs = [(name, vectors[i]) for i, (name, _) in enumerate(jd_files)]
        
        self._jd_files_cache = jd_files
        self._jd_embs_cache = jd_embs
        return jd_files, jd_embs

    def match_resume_with_jds(self, user_id: str, threshold: float = 0.65) -> List[Dict[str, Any]]:
        """Match user's resume with job descriptions using fast similarity pre-filtering and parallel AI analysis"""
        if not self.openai_configured:
            return [{
                "error": "AI provider not configured. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in backend/.env"
            }]
            
        try:
            # Get resume content
            resume_content = self.get_resume_content(user_id)
            if not resume_content:
                print("[WARN] No resume content found for user")
                return []
            
            print(f"[INFO] Resume content length: {len(resume_content)} characters")
            
            jd_files, jd_embs = self._get_jd_embeddings_and_files()
            if not jd_files:
                print("[WARN] No JD files found to match against")
                return []
            
            # Embed only the resume (1 fast API call)
            resume_emb = self.embeddings.embed_query(resume_content)
            
            # Calculate cosine similarity
            def cosine_sim(a, b):
                a_arr = np.array(a)
                b_arr = np.array(b)
                denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
                if denom == 0:
                    return 0.0
                return float(np.dot(a_arr, b_arr) / denom)
            
            # Score all JDs
            scored_jds = []
            for (jd_name, jd_text), (_, jd_emb) in zip(jd_files, jd_embs):
                sim = cosine_sim(resume_emb, jd_emb)
                scored_jds.append((jd_name, jd_text, sim))
            
            # Sort by cosine similarity descending
            scored_jds.sort(key=lambda x: x[2], reverse=True)
            
            for name, _, score in scored_jds:
                print(f"[INFO] {name}: Cosine similarity = {score:.3f}")
            
            # Pick top candidates for deep AI analysis (top 4 highest or all with score >= 0.70)
            top_candidates = [item for item in scored_jds if item[2] >= 0.70]
            if len(top_candidates) < 3:
                top_candidates = scored_jds[:4]
            else:
                top_candidates = top_candidates[:5]
                
            print(f"[INFO] Running parallel AI analysis on top {len(top_candidates)} candidates...")
            
            def _analyze_single(jd_item):
                jd_name, jd_text, similarity_score = jd_item
                try:
                    chain = self.matching_prompt | self.model
                    response = chain.invoke({
                        "resume_content": resume_content,
                        "jd_content": jd_text
                    })
                    raw_content = response.content.strip()
                    
                    # Clean markdown code blocks and extract JSON
                    start_idx = raw_content.find('{')
                    end_idx = raw_content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = raw_content[start_idx:end_idx+1]
                    else:
                        json_str = raw_content
                    
                    result = json.loads(json_str)
                    
                    # Support match_score, match_percentage, or score fields
                    score_val = (
                        result.get("match_score")
                        or result.get("match_percentage")
                        or result.get("score")
                        or result.get("fit_score")
                        or 0
                    )
                    if isinstance(score_val, str):
                        try:
                            score_val = float(score_val.replace("%", "").strip())
                        except ValueError:
                            score_val = 0
                    
                    ai_match_percentage = float(score_val) / 100.0 if score_val > 1.0 else float(score_val)
                    
                    # Weight: 40% cosine similarity, 60% AI analysis
                    combined_score = (similarity_score * 0.4) + (ai_match_percentage * 0.6)
                    print(f"[INFO] {jd_name}: Cosine={similarity_score:.3f}, AI={ai_match_percentage:.3f}, Combined={combined_score:.3f}")
                    
                    if combined_score >= threshold:
                        question_file = _map_jd_name_to_question_file(jd_name)
                        if not question_file:
                            question_file = f"{jd_name.lower().replace(' ', '_')}.txt"
                        
                        return {
                            "jd_title": jd_name,
                            "jd_source": question_file,
                            "match_percentage": combined_score,
                            "cosine_similarity": similarity_score,
                            "ai_score": ai_match_percentage,
                            "analysis": result.get("analysis", ""),
                            "key_matches": result.get("key_matches", []),
                            "missing_skills": result.get("missing_skills", []),
                            "recommendations": result.get("recommendations", "")
                        }
                    else:
                        print(f"[INFO] Below threshold: {jd_name} - {combined_score * 100:.1f}% (threshold {threshold*100:.1f}%)")
                        return None
                except Exception as ex:
                    print(f"[ERROR] Error analyzing {jd_name}: {ex}")
                    return None
            
            matches = []
            with ThreadPoolExecutor(max_workers=min(len(top_candidates), 5)) as executor:
                futures = [executor.submit(_analyze_single, item) for item in top_candidates]
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        matches.append(res)
            
            # Sort matches highest first
            matches.sort(key=lambda x: x["match_percentage"], reverse=True)
            print(f"[INFO] Found {len(matches)} matches above {threshold * 100:.1f}% threshold")
            return matches
            
        except Exception as e:
            print(f"[ERROR] Error in resume-JD matching: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_jd_content(self, jd_filename: str) -> str:
        """Get content of a specific job description"""
        try:
            jd_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "jd")
            jd_path = os.path.join(jd_dir, jd_filename)
            
            if os.path.exists(jd_path):
                with open(jd_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return ""
                
        except Exception as e:
            print(f"Error reading JD file: {e}")
            return ""

    def score_transcript(self, transcript: str) -> int:
        """Evaluate a transcript using OpenAI and return a score (percentage out of 100)."""
        if not self.openai_configured or not self.model:
            return 0
        prompt = f"""
        You are an expert technical interviewer. Evaluate the following interview transcript and give a strict score out of 100 based on the quality, relevance, and completeness of the answers. Only return a number (integer, 0-100).
        
        TRANSCRIPT:
        {transcript}
        """
        try:
            response = self.model.invoke(prompt)
            # Extract integer score from response
            score = int(''.join(filter(str.isdigit, str(response.content))))
            return min(max(score, 0), 100)
        except Exception as e:
            print(f"Error scoring transcript: {e}")
            return 0

# Initialize the service
rag_service = RAGService() 