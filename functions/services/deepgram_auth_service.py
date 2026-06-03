# Stub service to prevent import errors
# The deepgram_auth_service is not deployed on Render

class DeepgramAuthService:
    def __init__(self):
        self._api_key = None
    
    async def get_api_key(self) -> str:
        """Get Deepgram API key (stub for local/dev only)."""
        return "stub-api-key"
    
    async def validate_key(self, key: str) -> bool:
        """Validate Deepgram API key (stub for local/dev only)."""
        return True

deepgram_auth_service = DeepgramAuthService()