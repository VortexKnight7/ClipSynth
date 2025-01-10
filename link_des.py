# import yt_dlp
# from datetime import timedelta

# def get_video_metadata(youtube_url):
#     try:
#         # Define options for yt-dlp
#         ydl_opts = {
#             'quiet': True,  # Suppress verbose output
#             'skip_download': True,  # Do not download the video
#         }

#         # Use yt-dlp to extract video info
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             print(f"DEBUG: Fetching metadata for URL: {youtube_url}")
#             info = ydl.extract_info(youtube_url, download=False)

#         # Extract highest resolution stream info
#         formats = info.get("formats", [])
#         best_stream = max(
#             (fmt for fmt in formats if fmt.get("vcodec") != "none"), 
#             key=lambda x: x.get("height", 0),
#             default=None
#         )

#         highest_resolution = f"{best_stream['height']}p" if best_stream else "Unknown"
#         file_size = (
#             round(best_stream["filesize"] / (1024 * 1024), 2)
#             if best_stream and best_stream.get("filesize")
#             else "Unknown"
#         )

#         # Extract relevant metadata
#         video_metadata = {
#             "Duration (seconds)": info.get("duration", 0),
#             "Duration (HH:MM:SS)": str(timedelta(seconds=info.get("duration", 0))),
#             "Highest Resolution": highest_resolution,
#             "File Size (MB)": file_size,
#         }

#         print("DEBUG: Successfully fetched metadata.")
#         return video_metadata

#     except Exception as e:
#         print(f"DEBUG: Error occurred - {str(e)}")
#         return {"Error": str(e)}


# # Example Usage
# if __name__ == "__main__":
#     youtube_url = input("Enter the YouTube video URL: ")
#     print("DEBUG: Starting metadata extraction process.")
#     metadata = get_video_metadata(youtube_url)
#     print("DEBUG: Metadata extraction process complete.")
#     for key, value in metadata.items():
#         if isinstance(value, list):  # Handle list values like tags
#             print(f"{key}: {', '.join(value)}")
#         else:
#             print(f"{key}: {value}")




import yt_dlp
from datetime import timedelta

def get_video_metadata(youtube_url):
    try:
        # Define options for yt-dlp
        ydl_opts = {
            'quiet': True,  # Suppress verbose output
            'skip_download': True,  # Do not download the video
        }

        # Use yt-dlp to extract video info
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"DEBUG: Fetching metadata for URL: {youtube_url}")
            info = ydl.extract_info(youtube_url, download=False)

        # Extract highest resolution stream info
        formats = info.get("formats", [])
        best_stream = max(
            (fmt for fmt in formats if fmt.get("vcodec") != "none"), 
            key=lambda x: x.get("height", 0),
            default=None
        )

        highest_resolution = f"{best_stream['height']}p" if best_stream else "Unknown"

        # Calculate file size (fallback if filesize is not available)
        if best_stream and best_stream.get("filesize"):
            file_size = round(best_stream["filesize"] / (1024 * 1024), 2)
        elif best_stream and "tbr" in best_stream and info.get("duration"):
            # Estimate file size using average bitrate (tbr) and duration
            bitrate_kbps = best_stream["tbr"]  # Average bitrate in kbps
            duration_sec = info.get("duration", 0)  # Duration in seconds
            file_size = round((bitrate_kbps * duration_sec * 0.125) / (1024), 2)  # File size in MB
        else:
            file_size = "Unknown"

        # Extract relevant metadata
        video_metadata = {
            "Duration (seconds)": info.get("duration", 0),
            "Duration (HH:MM:SS)": str(timedelta(seconds=info.get("duration", 0))),
            "Highest Resolution": highest_resolution,
            "File Size (MB)": file_size,
        }

        print("DEBUG: Successfully fetched metadata.")
        return video_metadata

    except Exception as e:
        print(f"DEBUG: Error occurred - {str(e)}")
        return {"Error": str(e)}


# Example Usage
if __name__ == "__main__":
    youtube_url = input("Enter the YouTube video URL: ")
    print("DEBUG: Starting metadata extraction process.")
    metadata = get_video_metadata(youtube_url)
    print("DEBUG: Metadata extraction process complete.")
    for key, value in metadata.items():
        if isinstance(value, list):  # Handle list values like tags
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
