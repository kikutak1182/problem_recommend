import requests
import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, urlparse
from datetime import datetime

from crawler_config import config

class EditorialExtractor:
    """Extract editorial URLs from AtCoder contest pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent
        })
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for crawler operations"""
        logger = logging.getLogger('editorial_extractor')
        logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        if not logger.handlers:
            handler = logging.FileHandler(config.log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_contest_editorial_page(self, contest_id: str) -> Optional[str]:
        """Get contest editorial page URL with Japanese language parameter"""
        base_url = f"https://atcoder.jp/contests/{contest_id}/editorial?lang={config.editorial_lang}"
        
        try:
            response = self._make_request(base_url)
            if response and response.status_code == 200:
                return base_url
        except Exception as e:
            self.logger.error(f"Failed to access editorial page for {contest_id}: {e}")
        
        return None
    
    def extract_problem_editorial_urls(self, contest_id: str) -> Dict[str, Dict]:
        """
        Extract editorial URLs for all problems in a contest
        
        Returns:
            Dict mapping problem_index to editorial info
        """
        editorial_page_url = self.extract_contest_editorial_page(contest_id)
        if not editorial_page_url:
            return {}
        
        try:
            response = self._make_request(editorial_page_url)
            if not response or response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_editorial_links(soup, contest_id)
            
        except Exception as e:
            self.logger.error(f"Failed to extract editorial URLs for {contest_id}: {e}")
            return {}
    
    def _parse_editorial_links(self, soup: BeautifulSoup, contest_id: str) -> Dict[str, Dict]:
        """Parse editorial links from contest editorial page"""
        editorial_mappings = {}
        
        # Find all h3 and h4 headings that match problem pattern
        headings = soup.find_all(['h3', 'h4'])
        
        for heading in headings:
            problem_match = self._extract_problem_index_from_heading(heading)
            if not problem_match:
                continue
            
            problem_index = problem_match
            editorial_url = self._find_editorial_link_in_section(heading, contest_id)
            
            if editorial_url:
                # Editorial ID extracted from Japanese page is already Japanese version
                # No need to add language parameter to individual editorial URLs
                pass
                
                editorial_id = self._extract_editorial_id(editorial_url)
                
                editorial_mappings[problem_index] = {
                    "contest_id": contest_id,
                    "problem_index": problem_index,
                    "problem_url": f"https://atcoder.jp/contests/{contest_id}/tasks/{contest_id}_{problem_index.lower()}",
                    "editorial_url": editorial_url,
                    "editorial_id": editorial_id,
                    "extracted_at": datetime.now().isoformat(),
                    "status": "extracted"
                }
                
                self.logger.info(f"Found editorial for {contest_id} {problem_index}: {editorial_url}")
        
        return editorial_mappings
    
    def _extract_problem_index_from_heading(self, heading: Tag) -> Optional[str]:
        """Extract problem index (A, B, C, Ex, etc.) from heading text"""
        if not heading or not heading.text:
            return None
        
        heading_text = heading.text.strip()
        
        # Match pattern: ^(Ex|[A-Z])\s*-
        match = re.match(r'^(Ex|[A-Z])\s*-', heading_text)
        if match:
            return match.group(1)
        
        return None
    
    def _find_editorial_link_in_section(self, heading: Tag, contest_id: str) -> Optional[str]:
        """Find editorial link in the same section as the heading"""
        # Look for links in the same section (until next heading or end)
        current = heading.next_sibling
        
        while current:
            # Stop if we hit another heading of same or higher level
            if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4']:
                break
            
            # Look for links in current element and its descendants
            if hasattr(current, 'find_all'):
                links = current.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    if self._is_valid_editorial_link(href, contest_id):
                        return urljoin("https://atcoder.jp", href)
            
            current = current.next_sibling
        
        return None
    
    def _is_valid_editorial_link(self, href: str, contest_id: str) -> bool:
        """Check if href is a valid editorial link"""
        # Pattern: ^/contests/<contest-id>/editorial/\d+/?$
        pattern = rf'^/contests/{re.escape(contest_id)}/editorial/\d+/?$'
        return bool(re.match(pattern, href))
    
    def _extract_editorial_id(self, editorial_url: str) -> Optional[int]:
        """Extract editorial ID from editorial URL"""
        try:
            # Extract ID from path: /contests/{contest}/editorial/{id}
            path = urlparse(editorial_url).path
            editorial_id = int(path.rstrip('/').split('/')[-1])
            return editorial_id
        except (ValueError, IndexError):
            return None
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(config.max_retries):
            try:
                self.logger.info(f"Requesting: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=config.timeout)
                
                # Add delay between requests
                time.sleep(config.request_delay)
                
                return response
                
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    self.logger.error(f"All retry attempts failed for {url}")
        
        return None
    
    def validate_editorial_url(self, editorial_url: str) -> bool:
        """Validate if editorial URL is accessible"""
        try:
            response = self._make_request(editorial_url)
            return response is not None and response.status_code == 200
        except Exception:
            return False