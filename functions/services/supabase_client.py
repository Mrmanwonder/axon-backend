from supabase import create_client, Client

SUPABASE_URL = "https://anmfwzxyvqxyxxeobxti.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_fIbfGtT5yyFaogq4DQAuxw_tZ54kolM"

def get_supabase_client() -> Client:
    """Create Supabase client with anon key for public queries."""
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)