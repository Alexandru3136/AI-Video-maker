"""Optional YouTube upload via Google API.

Requires:
  pip install google-api-python-client google-auth-oauthlib
  A client_secrets.json OAuth2 file from Google Cloud Console.
  Set YOUTUBE_CLIENT_SECRETS in .env to its path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if YouTube upload dependencies and credentials are configured."""
    try:
        import google_auth_oauthlib  # noqa: F401
        import googleapiclient  # noqa: F401
    except ImportError:
        return False
    secrets = os.getenv("YOUTUBE_CLIENT_SECRETS", "").strip()
    return bool(secrets) and Path(secrets).is_file()


def upload_video(
    video_path: Path,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    privacy: str = "private",
) -> str:
    """Upload a video to YouTube. Returns the video ID.

    privacy: 'private', 'unlisted', or 'public'.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    secrets_path = os.getenv("YOUTUBE_CLIENT_SECRETS", "")
    token_path = Path(secrets_path).parent / "youtube_token.json"
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]

    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(secrets_path, scopes)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": (tags or [])[:30],
            "categoryId": "22",
        },
        "status": {"privacyStatus": privacy},
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("YouTube upload progress: %d%%", int(status.progress() * 100))

    video_id = response["id"]
    log.info("YouTube upload complete: https://youtu.be/%s", video_id)
    return video_id
