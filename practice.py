import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(".env")

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

response = (
    supabase.table("todos")
    .delete()
    .eq("id", 2)
    .execute()
)

print(response)