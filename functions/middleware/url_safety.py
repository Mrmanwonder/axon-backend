# Stub middleware to prevent import errors
# The url_safety middleware is not deployed on Render


def assert_safe_https_url(url: str) -> None:
    """Assert URL is HTTPS (stub for local/dev only)."""
    if not url.startswith("https://") and not url.startswith("http://localhost"):
        raise ValueError(f"URL must be HTTPS: {url}")
