"""Debug helper: print Wipro chunks that mention key P&L terms."""
import json

from config import settings

keywords = [
    "revenue from operations",
    "total income",
    "consolidated statement of profit and loss",
]

# settings.chunks_path is absolute, so this works from any folder.
with settings.chunks_path.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        text = chunk.get("text", "")
        text_lower = text.lower()

        if (
            chunk.get("company", "").lower() == "wipro"
            and any(keyword in text_lower for keyword in keywords)
        ):
            print("\n" + "=" * 80)
            print("Page:", chunk.get("page_number"))
            print("Section:", chunk.get("section"))
            print("Chunk ID:", chunk.get("chunk_id"))
            print(text)
