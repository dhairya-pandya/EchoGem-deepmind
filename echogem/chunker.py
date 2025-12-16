"""
Transcript chunking module using Google Gemini for intelligent segmentation.
"""

import os
import json
import re
import google.generativeai as genai
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .models import Chunk


class Chunker:
    """
    Intelligent transcript chunking using LLM-based semantic analysis
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        max_tokens: int = 2000,
        similarity_threshold: float = 0.82,
        coherence_threshold: float = 0.75,
    ):
        """
        Initialize the chunker
        
        Args:
            api_key: Google API key for Gemini
            embed_model: Path to sentence transformer model or model name
            max_tokens: Maximum tokens per chunk
            similarity_threshold: Threshold for semantic similarity
            coherence_threshold: Threshold for coherence
        """
        # Initialize Google Gemini
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize sentence transformer
        try:
            self.embedder = SentenceTransformer(embed_model)
        except Exception as e:
            print(f"Warning: Could not load model {embed_model}: {e}")
            print("Using default model")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.max_tokens = max_tokens
        self.sim_threshold = similarity_threshold
        self.coh_threshold = coherence_threshold

    def load_transcript(self, file_path: str) -> str:
        """
        Load transcript text from file
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Transcript text content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()
            print(f"Transcript loaded ({len(transcript)} characters)")
            return transcript
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        """
        Chunk transcript using LLM-based semantic analysis
        
        Args:
            transcript: Transcript text to chunk
            
        Returns:
            List of Chunk objects
        """
        try:
            # Create chunking prompt
            prompt = self._create_prompt(transcript)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse response
            chunks = self._parse_chunk_response(response.text)
            
            print(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error during chunking: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(transcript)

    def _create_prompt(self, transcript: str) -> str:
        """Create the chunking prompt"""
        return f"""
        **SYSTEM PROMPT**
        You are a transcript processing expert. The following transcript needs to be chunked very intelligently and logically. Ensure sensible segments and structure to be later provided as context to answer questions.

        **INSTRUCTIONS**
        1. Create as many or as few chunks as needed
        2. Each chunk should contain consecutive sentences
        3. For each chunk provide:
          - title: 2-5 word summary
          - content: exact sentences
          - keywords: 3-5 important terms
          - named_entities: any mentioned names
          - timestamp_range: estimate like "00:00-01:30"

        **TRANSCRIPT**
        {transcript[:5000]}...

        **OUTPUT FORMAT**
        You must output ONLY valid JSON in this exact format:
        {{
          "chunks": [
            {{
              "title": "Summary",
              "content": "Actual sentences",
              "keywords": ["term1", "term2"],
              "named_entities": ["Name"],
              "timestamp_range": "00:00-01:30"
            }}
          ]
        }}
        """

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response, handling various formats"""
        # Try markdown code blocks first
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        if matches:
            json_str = max(matches, key=len).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Find JSON with balanced braces
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_str = response_text[start_idx:i + 1]
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        start_idx = -1
                        brace_count = 0
                        continue
        
        # Fallback: find first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response_text[start_idx:end_idx]
        json.loads(json_str)
        return json_str

    def _parse_chunk_response(self, response_text: str) -> List[Chunk]:
        """Parse the LLM response into Chunk objects"""
        try:
            json_str = self._extract_json_from_response(response_text)
            data = json.loads(json_str)
            
            chunks = []
            for chunk_data in data.get('chunks', []):
                chunk = Chunk(
                    title=chunk_data.get('title', 'Untitled'),
                    content=chunk_data.get('content', ''),
                    keywords=chunk_data.get('keywords', []),
                    named_entities=chunk_data.get('named_entities', []),
                    timestamp_range=chunk_data.get('timestamp_range', ''),
                    chunk_id=f"chunk_{len(chunks)}"
                )
                chunks.append(chunk)
            
            return chunks
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing chunk response: {e}")
            return []
        except Exception as e:
            print(f"Error parsing chunk response: {e}")
            return []

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\s*$)'
        sentences = re.split(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        if not cleaned_sentences:
            return [text.strip()] if text.strip() else []
        
        return cleaned_sentences

    def _group_sentences_into_chunks(
        self, 
        sentences: List[str], 
        max_words: int,
        flexibility: int = 200
    ) -> List[str]:
        """Group sentences into chunks respecting max_words limit"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        min_words = max(1, max_words - flexibility)
        max_words_actual = max_words + flexibility
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > max_words_actual and current_chunk:
                if current_word_count >= min_words:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _fallback_chunking(self, transcript: str) -> List[Chunk]:
        """Fallback chunking method using sentence-aware text splitting"""
        if not transcript.strip():
            return []
        
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n+', transcript)
        
        # Split paragraphs into sentences
        all_sentences = []
        for para in paragraphs:
            para = para.strip()
            if para:
                sentences = self._split_into_sentences(para)
                all_sentences.extend(sentences)
        
        # Fallback to basic sentence splitting
        if not all_sentences:
            all_sentences = [s.strip() for s in re.split(r'[.!?]+\s+', transcript) if s.strip()]
        
        # Ultimate fallback: word-based chunking
        if not all_sentences:
            words = transcript.split()
            chunks = []
            chunk_size = self.max_tokens
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk = Chunk(
                    title=f"Chunk {len(chunks) + 1}",
                    content=chunk_text,
                    keywords=[],
                    named_entities=[],
                    timestamp_range="",
                    chunk_id=f"fallback_chunk_{len(chunks)}"
                )
                chunks.append(chunk)
            
            return chunks
        
        # Group sentences into chunks
        chunk_texts = self._group_sentences_into_chunks(all_sentences, self.max_tokens)
        
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            first_words = chunk_text.split()[:5]
            title = ' '.join(first_words)
            if len(title) > 50:
                title = title[:47] + "..."
            
            chunk = Chunk(
                title=title if title else f"Chunk {i + 1}",
                content=chunk_text,
                keywords=[],
                named_entities=[],
                timestamp_range="",
                chunk_id=f"fallback_chunk_{i}"
            )
            chunks.append(chunk)
        
        return chunks

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using sentence transformer"""
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return [0.0] * 384  # Default dimension
