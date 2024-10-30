"""
Handles paper recommendations and academic database integration.
Provides functionality to fetch and recommend related papers based on conversation
context and document content using arXiv and other academic sources.
"""
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
import re
from dataclasses import dataclass
import arxiv
import requests
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecommendedPaper:
    """Represents a recommended paper with its metadata."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: Optional[datetime]
    source: str  # e.g., 'arxiv', 'semantic_scholar'
    score: float  # relevance score
    citations: Optional[int] = None
    pdf_url: Optional[str] = None
    keywords: Optional[List[str]] = None

class PaperRecommender:
    """Manages paper recommendations from various academic sources."""
    
    def __init__(
        self,
        semantic_scholar_api_key: Optional[str] = None,
        max_results: int = 5,
        cache_duration_hours: int = 24
    ):
        """
        Initialize the paper recommender.
        
        Args:
            semantic_scholar_api_key: Optional API key for Semantic Scholar
            max_results: Maximum number of recommendations to return
            cache_duration_hours: How long to cache results
        """
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.max_results = max_results
        self.cache_duration_hours = cache_duration_hours
        
        # Initialize arXiv client with reasonable defaults
        self.arxiv_client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=3
        )
        
        # Initialize result cache
        self._cache: Dict[str, Dict] = {}
        
    def get_recommendations(
        self,
        context: Dict,
        filter_criteria: Optional[Dict] = None
    ) -> List[RecommendedPaper]:
        """
        Get paper recommendations based on conversation context.
        
        Args:
            context: Dictionary containing conversation context including:
                    - current_topic: str
                    - keywords: List[str]
                    - recent_papers: List[str] (titles/ids of papers being discussed)
            filter_criteria: Optional filters for recommendations (e.g., date range, 
                           subject areas)
                           
        Returns:
            List of RecommendedPaper objects sorted by relevance
            
        Raises:
            ValueError: If context is invalid
            RuntimeError: If recommendation retrieval fails
        """
        if not context.get('current_topic') and not context.get('keywords'):
            raise ValueError("Context must contain either current_topic or keywords")
            
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context, filter_criteria)
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                return cached_results
            
            # Get recommendations from multiple sources
            arxiv_papers = self._get_arxiv_recommendations(context, filter_criteria)
            semantic_papers = self._get_semantic_scholar_recommendations(
                context, filter_criteria
            ) if self.semantic_scholar_api_key else []
            
            # Merge and rank recommendations
            all_papers = self._merge_and_rank_recommendations(
                arxiv_papers + semantic_papers
            )
            
            # Cache results
            self._add_to_cache(cache_key, all_papers)
            
            return all_papers[:self.max_results]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise RuntimeError(f"Failed to get recommendations: {str(e)}") from e
            
    def _get_arxiv_recommendations(
        self,
        context: Dict,
        filter_criteria: Optional[Dict]
    ) -> List[RecommendedPaper]:
        """Get recommendations from arXiv."""
        # Construct arXiv search query
        query_parts = []
        
        if context.get('current_topic'):
            query_parts.append(context['current_topic'])
            
        if context.get('keywords'):
            query_parts.extend(context['keywords'])
            
        if filter_criteria and filter_criteria.get('date_range'):
            date_range = filter_criteria['date_range']
            if date_range.get('start'):
                query_parts.append(f"submittedDate:[{date_range['start']} TO *]")
                
        query = ' AND '.join(f'"{part}"' for part in query_parts)
        
        # Search arXiv
        search = arxiv.Search(
            query=query,
            max_results=self.max_results * 2,  # Get extra for ranking
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in self.arxiv_client.results(search):
            # Convert to RecommendedPaper format
            recommended = RecommendedPaper(
                title=paper.title,
                authors=[str(author) for author in paper.authors],
                abstract=paper.summary,
                url=paper.entry_id,
                published_date=paper.published,
                source='arxiv',
                score=1.0,  # Base score, will be adjusted during ranking
                pdf_url=paper.pdf_url,
                keywords=self._extract_keywords(paper.summary)
            )
            results.append(recommended)
            
        return results
        
    def _get_semantic_scholar_recommendations(
        self,
        context: Dict,
        filter_criteria: Optional[Dict]
    ) -> List[RecommendedPaper]:
        """Get recommendations from Semantic Scholar."""
        if not self.semantic_scholar_api_key:
            return []
            
        base_url = "https://api.semanticscholar.org/graph/v1"
        headers = {"x-api-key": self.semantic_scholar_api_key}
        
        # Construct search query
        query = context.get('current_topic', '') + ' ' + \
                ' '.join(context.get('keywords', []))
        
        try:
            # Search papers
            response = requests.get(
                f"{base_url}/paper/search",
                params={
                    "query": query,
                    "limit": self.max_results * 2,
                    "fields": "title,authors,abstract,url,year,citations,references"
                },
                headers=headers
            )
            response.raise_for_status()
            papers = response.json().get('data', [])
            
            results = []
            for paper in papers:
                if not paper.get('title') or not paper.get('abstract'):
                    continue
                    
                recommended = RecommendedPaper(
                    title=paper['title'],
                    authors=[author.get('name', '') for author in paper.get('authors', [])],
                    abstract=paper.get('abstract', ''),
                    url=paper.get('url', ''),
                    published_date=datetime(int(paper.get('year', 2020)), 1, 1)
                    if paper.get('year') else None,
                    source='semantic_scholar',
                    score=1.0,  # Base score, will be adjusted during ranking
                    citations=paper.get('citations', {}).get('total', 0),
                    keywords=self._extract_keywords(paper.get('abstract', ''))
                )
                results.append(recommended)
                
            return results
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching from Semantic Scholar: {str(e)}")
            return []
            
    def _merge_and_rank_recommendations(
        self,
        papers: List[RecommendedPaper]
    ) -> List[RecommendedPaper]:
        """
        Merge and rank recommendations from different sources.
        Uses a scoring system based on relevance, citations, and recency.
        """
        # Remove duplicates (based on title similarity)
        unique_papers = self._remove_duplicates(papers)
        
        # Score papers
        for paper in unique_papers:
            # Base score (1.0) adjusted by factors:
            
            # Citation score (if available)
            citation_score = 0.0
            if paper.citations is not None:
                citation_score = min(paper.citations / 1000, 1.0)  # Cap at 1.0
                
            # Recency score
            recency_score = 0.0
            if paper.published_date:
                years_old = (datetime.now() - paper.published_date).days / 365
                recency_score = max(0, 1 - (years_old / 5))  # Linear decay over 5 years
                
            # Keyword relevance score
            keyword_score = len(paper.keywords) / 10 if paper.keywords else 0.0
            
            # Combined score with weights
            paper.score = (
                0.4 +  # Base relevance from search
                0.3 * citation_score +
                0.2 * recency_score +
                0.1 * keyword_score
            )
            
        # Sort by score
        return sorted(unique_papers, key=lambda x: x.score, reverse=True)
        
    def _remove_duplicates(
        self,
        papers: List[RecommendedPaper]
    ) -> List[RecommendedPaper]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            norm_title = self._normalize_title(paper.title)
            
            # Check if we've seen a similar title
            if not any(self._titles_similar(norm_title, seen) for seen in seen_titles):
                unique_papers.append(paper)
                seen_titles.add(norm_title)
                
        return unique_papers
        
    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for comparison."""
        return re.sub(r'[^\w\s]', '', title.lower())
        
    @staticmethod
    def _titles_similar(title1: str, title2: str) -> bool:
        """Check if titles are similar using basic fuzzy matching."""
        # Simple Levenshtein distance ratio
        from difflib import SequenceMatcher
        return SequenceMatcher(None, title1, title2).ratio() > 0.8
        
    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract potential keywords from text."""
        # Simple keyword extraction based on common patterns
        keywords = []
        
        # Look for phrases in quotes
        keywords.extend(re.findall(r'"([^"]+)"', text))
        
        # Look for capitalized phrases
        keywords.extend(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text))
        
        # Look for common technical terms
        technical_terms = re.findall(
            r'\b(?:CNN|RNN|LSTM|GAN|transformer|neural|network|deep\s+learning|machine\s+learning|AI)\b',
            text,
            re.IGNORECASE
        )
        keywords.extend(technical_terms)
        
        return list(set(keywords))  # Remove duplicates
        
    def _generate_cache_key(self, context: Dict, filter_criteria: Optional[Dict]) -> str:
        """Generate a cache key for the given context and filters."""
        key_parts = [
            context.get('current_topic', ''),
            ','.join(context.get('keywords', [])),
            str(filter_criteria)
        ]
        return '_'.join(key_parts)
        
    def _get_from_cache(self, cache_key: str) -> Optional[List[RecommendedPaper]]:
        """Get recommendations from cache if available and not expired."""
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            cache_age = (datetime.now() - cache_entry['timestamp']).total_seconds() / 3600
            
            if cache_age < self.cache_duration_hours:
                return cache_entry['results']
                
        return None
        
    def _add_to_cache(self, cache_key: str, results: List[RecommendedPaper]):
        """Add results to cache."""
        self._cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now()
        }
        
        # Clean up old cache entries
        self._cleanup_cache()
        
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            age = (current_time - entry['timestamp']).total_seconds() / 3600
            if age >= self.cache_duration_hours:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self._cache[key]