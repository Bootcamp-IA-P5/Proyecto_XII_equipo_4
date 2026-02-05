"""
Video Downloader Module - Downloads videos from YouTube, Instagram, TikTok, and Facebook
Handles URL validation and file management
"""

import os
import requests
from pathlib import Path
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse
import tempfile
import logging

logger = logging.getLogger(__name__)

# Optional imports for video downloading
try:
    from yt_dlp import YoutubeDL
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False

try:
    import instagrapi
    HAS_INSTAGRAPI = True
except ImportError:
    HAS_INSTAGRAPI = False


class VideoDownloader:
    """Download videos from various social media platforms."""
    
    SUPPORTED_PLATFORMS = {
        'youtube': ['youtube.com', 'youtu.be', 'youtube-nocookie.com'],
        'instagram': ['instagram.com', 'instagr.am'],
        'tiktok': ['tiktok.com', 'vm.tiktok.com'],
        'facebook': ['facebook.com', 'fb.com', 'facebook.watch'],
        'twitter': ['twitter.com', 'x.com'],
    }
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the video downloader.
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "video_downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def detect_platform(url: str) -> Optional[str]:
        """
        Detect which platform a URL belongs to.
        
        Args:
            url: Video URL
            
        Returns:
            Platform name or None if not supported
        """
        parsed = urlparse(url.lower())
        domain = parsed.netloc.replace('www.', '')
        
        for platform, domains in VideoDownloader.SUPPORTED_PLATFORMS.items():
            if any(d in domain for d in domains):
                return platform
        
        return None
    
    def download_youtube(
        self,
        url: str,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Download video from YouTube.
        
        Args:
            url: YouTube URL
            video_name: Optional custom name for the video
            
        Returns:
            Dictionary with status, path, and metadata
        """
        if not HAS_YT_DLP:
            return {
                'success': False,
                'error': 'yt-dlp not installed. Run: pip install yt-dlp'
            }
        
        try:
            output_template = str(self.output_dir / (video_name or '%(title)s')) + '.%(ext)s'
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading YouTube video: {url}")
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
                
                return {
                    'success': True,
                    'path': file_path,
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'platform': 'youtube'
                }
        except Exception as e:
            logger.error(f"Error downloading YouTube video: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'youtube'
            }
    
    def download_instagram(
        self,
        url: str,
        video_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict:
        """
        Download video from Instagram (requires credentials).
        
        Args:
            url: Instagram URL
            video_name: Optional custom name
            username: Instagram username
            password: Instagram password
            
        Returns:
            Dictionary with status and path
        """
        if not HAS_INSTAGRAPI:
            return {
                'success': False,
                'error': 'instagrapi not installed. Run: pip install instagrapi',
                'platform': 'instagram'
            }
        
        if not username or not password:
            return {
                'success': False,
                'error': 'Instagram login credentials required',
                'platform': 'instagram'
            }
        
        try:
            from instagrapi import Client
            
            client = Client()
            client.login(username, password)
            
            # Extract media ID from URL
            if 'instagram.com/p/' in url:
                media_id = url.split('/p/')[1].rstrip('/')
            elif 'instagram.com/reel/' in url:
                media_id = url.split('/reel/')[1].rstrip('/')
            else:
                return {
                    'success': False,
                    'error': 'Could not extract media ID from Instagram URL',
                    'platform': 'instagram'
                }
            
            # Download media
            media = client.media_info(int(media_id)).dict()
            
            if media.get('video_url'):
                video_url = media['video_url']
                response = requests.get(video_url, stream=True)
                response.raise_for_status()
                
                # Save video
                file_name = video_name or f"instagram_{media_id}.mp4"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return {
                    'success': True,
                    'path': str(file_path),
                    'platform': 'instagram',
                    'media_id': media_id
                }
            else:
                return {
                    'success': False,
                    'error': 'Media is not a video or is not accessible',
                    'platform': 'instagram'
                }
        
        except Exception as e:
            logger.error(f"Error downloading Instagram video: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'instagram'
            }
    
    def download_tiktok(
        self,
        url: str,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Download video from TikTok.
        
        Args:
            url: TikTok URL
            video_name: Optional custom name
            
        Returns:
            Dictionary with status and path
        """
        if not HAS_YT_DLP:
            return {
                'success': False,
                'error': 'yt-dlp not installed. Run: pip install yt-dlp',
                'platform': 'tiktok'
            }
        
        try:
            output_template = str(self.output_dir / (video_name or '%(title)s')) + '.%(ext)s'
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading TikTok video: {url}")
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
                
                return {
                    'success': True,
                    'path': file_path,
                    'platform': 'tiktok',
                    'uploader': info.get('uploader', 'Unknown'),
                }
        except Exception as e:
            logger.error(f"Error downloading TikTok video: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'tiktok'
            }
    
    def download_facebook(
        self,
        url: str,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Download video from Facebook.
        
        Args:
            url: Facebook URL
            video_name: Optional custom name
            
        Returns:
            Dictionary with status and path
        """
        if not HAS_YT_DLP:
            return {
                'success': False,
                'error': 'yt-dlp not installed. Run: pip install yt-dlp',
                'platform': 'facebook'
            }
        
        try:
            output_template = str(self.output_dir / (video_name or '%(title)s')) + '.%(ext)s'
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading Facebook video: {url}")
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
                
                return {
                    'success': True,
                    'path': file_path,
                    'platform': 'facebook',
                }
        except Exception as e:
            logger.error(f"Error downloading Facebook video: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'facebook'
            }
    
    def download_twitter(
        self,
        url: str,
        video_name: Optional[str] = None
    ) -> Dict:
        """
        Download video from Twitter/X.
        
        Args:
            url: Twitter/X URL
            video_name: Optional custom name
            
        Returns:
            Dictionary with status and path
        """
        if not HAS_YT_DLP:
            return {
                'success': False,
                'error': 'yt-dlp not installed. Run: pip install yt-dlp',
                'platform': 'twitter'
            }
        
        try:
            output_template = str(self.output_dir / (video_name or '%(title)s')) + '.%(ext)s'
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading Twitter/X video: {url}")
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
                
                return {
                    'success': True,
                    'path': file_path,
                    'platform': 'twitter',
                }
        except Exception as e:
            logger.error(f"Error downloading Twitter/X video: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': 'twitter'
            }
    
    def download(
        self,
        url: str,
        video_name: Optional[str] = None,
        credentials: Optional[Dict] = None
    ) -> Dict:
        """
        Download video from any supported platform.
        
        Args:
            url: Video URL
            video_name: Optional custom name
            credentials: Dict with 'username' and 'password' for platforms requiring authentication
            
        Returns:
            Dictionary with download result
        """
        platform = self.detect_platform(url)
        
        if not platform:
            return {
                'success': False,
                'error': 'Platform not supported. Supported platforms: YouTube, Instagram, TikTok, Facebook, Twitter'
            }
        
        if platform == 'youtube':
            return self.download_youtube(url, video_name)
        elif platform == 'instagram':
            creds = credentials or {}
            return self.download_instagram(url, video_name, creds.get('username'), creds.get('password'))
        elif platform == 'tiktok':
            return self.download_tiktok(url, video_name)
        elif platform == 'facebook':
            return self.download_facebook(url, video_name)
        elif platform == 'twitter':
            return self.download_twitter(url, video_name)
        else:
            return {
                'success': False,
                'error': f'Downloading from {platform} not yet implemented'
            }
