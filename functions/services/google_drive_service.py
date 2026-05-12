"""Google Drive service for backend file downloads."""
import os
import io
from typing import Optional, List, Dict, Any
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

class GoogleDriveService:
    """Service for accessing Google Drive with service account credentials."""
    
    def __init__(self):
        self._credentials = None
        self._service = None
        
    def initialize(self) -> None:
        """Initialize the Drive service using environment credentials."""
        # Get private key from environment
        private_key = os.getenv('GOOGLE_PRIVATE_KEY')
        if not private_key:
            raise ValueError("GOOGLE_PRIVATE_KEY environment variable not set")
            
        client_email = os.getenv(
            'GOOGLE_SERVICE_ACCOUNT_EMAIL',
            'drive-sync-bot@axon-34b8c.iam.gserviceaccount.com'
        )
        
        # Build credentials dict
        credentials_info = {
            "type": "service_account",
            "project_id": os.getenv('GOOGLE_PROJECT_ID', 'axon-34b8c'),
            "private_key": private_key.replace('\\n', '\n'),
            "client_email": client_email,
            "client_id": None,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": None
        }
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        self._credentials = credentials
        self._service = build('drive', 'v3', credentials=credentials)
    
    def find_folder(self, folder_name: str) -> Optional[str]:
        """Find a folder by name and return its ID."""
        if not self._service:
            self.initialize()
            
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        results = self._service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=10
        ).execute()
        
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None
    
    def list_files(self, folder_id: str) -> List[Dict[str, Any]]:
        """List all files in a folder."""
        if not self._service:
            self.initialize()
            
        query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder'"
        results = self._service.files().list(
            q=query,
            fields="files(id, name, size)",
            pageSize=1000
        ).execute()
        
        files = []
        for f in results.get('files', []):
            files.append({
                'id': f['id'],
                'name': f['name'],
                'size': int(f.get('size', 0))
            })
        return files
    
    def download_file(self, file_id: str) -> io.BytesIO:
        """
        Download a file and return it as a BytesIO stream.
        Raises Exception on failure.
        """
        if not self._service:
            self.initialize()
            
        request = self._service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done:
            try:
                status, done = downloader.next_chunk()
            except Exception as e:
                raise Exception(f"Download failed: {e}")
                
        file_io.seek(0)
        return file_io
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get metadata for a file."""
        if not self._service:
            self.initialize()
            
        file = self._service.files().get(
            fileId=file_id,
            fields="id, name, size, mimeType"
        ).execute()
        
        return {
            'id': file.get('id'),
            'name': file.get('name'),
            'size': int(file.get('size', 0)),
            'mimeType': file.get('mimeType')
        }

# Singleton instance
drive_service = GoogleDriveService()
