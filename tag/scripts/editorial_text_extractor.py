import requests
import re
import time
import logging
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys
import os

# 設定ファイルをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config

class EditorialTextExtractor:
    """Extract text content from AtCoder editorial URLs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AtCoder Editorial Text Extractor (Educational Purpose)'
        })
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for text extraction operations"""
        logger = logging.getLogger('editorial_text_extractor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_editorial_text(self, editorial_url: str) -> Optional[str]:
        """
        Extract text content from editorial URL
        
        Args:
            editorial_url: AtCoder editorial URL
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            self.logger.info(f"Extracting text from: {editorial_url}")
            
            response = self._make_request(editorial_url)
            if not response or response.status_code != 200:
                self.logger.error(f"Failed to fetch {editorial_url}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            text_content = self._extract_main_content(soup)
            
            if not text_content.strip():
                self.logger.warning(f"No text content found in {editorial_url}")
                return None
            
            # Clean and format text
            cleaned_text = self._clean_text(text_content)
            
            self.logger.info(f"Successfully extracted {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {editorial_url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from editorial page"""
        
        # Try different content selectors in order of preference
        content_selectors = [
            '.editorial-content',
            '#main-container',
            '.post-content',
            '.content',
            'main',
            'article'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                return self._extract_text_from_element(content_div)
        
        # Fallback: extract from body, excluding navigation and footer
        body = soup.find('body')
        if body:
            # Remove navigation, footer, sidebar elements
            for tag in body.find_all(['nav', 'footer', 'aside', 'header']):
                tag.decompose()
            
            # Remove script and style tags
            for tag in body.find_all(['script', 'style']):
                tag.decompose()
                
            return self._extract_text_from_element(body)
        
        # Last resort: get all text
        return soup.get_text()
    
    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from BeautifulSoup element"""
        
        # Remove unwanted elements
        for tag in element.find_all(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()
        
        # Get text with proper spacing
        text_parts = []
        
        for string in element.stripped_strings:
            text_parts.append(string)
        
        return ' '.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common unwanted patterns
        patterns_to_remove = [
            r'AtCoder is a programming contest.*?',
            r'Sign up.*?Login',
            r'Cookie Policy.*?Privacy Policy',
            r'© AtCoder.*?',
            r'Powered by.*?',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize mathematical expressions
        text = re.sub(r'\\?\$([^$]+)\\?\$', r'(\1)', text)  # $formula$ -> (formula)
        text = re.sub(r'\\?\$\$([^$]+)\\?\$\$', r'(\1)', text)  # $$formula$$ -> (formula)
        
        # Clean up code blocks (keep content but mark as code)
        text = re.sub(r'```[a-zA-Z]*\n(.*?)\n```', r'[CODE: \1]', text, flags=re.DOTALL)
        
        # Remove URLs but keep the context
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        
        # Trim and clean final result
        text = text.strip()
        
        return text
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with proper delays"""
        try:
            response = self.session.get(url, timeout=30)
            
            # Add delay to be respectful
            time.sleep(1.0)
            
            return response
            
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
    
    def extract_multiple_editorials(self, editorial_urls: List[str]) -> Dict[str, str]:
        """
        Extract text from multiple editorial URLs
        
        Args:
            editorial_urls: List of editorial URLs
            
        Returns:
            Dict mapping URL to extracted text
        """
        results = {}
        
        for i, url in enumerate(editorial_urls, 1):
            self.logger.info(f"Processing {i}/{len(editorial_urls)}: {url}")
            
            text = self.extract_editorial_text(url)
            if text:
                results[url] = text
            
            # Progress feedback
            if i % 10 == 0:
                self.logger.info(f"Processed {i}/{len(editorial_urls)} editorials")
        
        self.logger.info(f"Successfully extracted {len(results)}/{len(editorial_urls)} editorials")
        return results

if __name__ == "__main__":
    # Test the extractor
    extractor = EditorialTextExtractor()
    
    # Test with a sample editorial URL
    test_url = "https://atcoder.jp/contests/abc175/editorial/51"
    
    print(f"Testing editorial text extraction with: {test_url}")
    text = extractor.extract_editorial_text(test_url)
    
    if text:
        print(f"Successfully extracted {len(text)} characters")
        print("First 500 characters:")
        print(text[:500])
        print("...")
    else:
        print("Failed to extract text")