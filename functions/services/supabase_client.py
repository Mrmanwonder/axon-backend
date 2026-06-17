import os
import requests

SUPABASE_URL = "https://anmfwzxyvqxyxxeobxti.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_fIbfGtT5yyFaogq4DQAuxw_tZ54kolM"


def get_supabase_client():
    """Simple Supabase client using requests."""
    class SimpleClient:
        def __init__(self, url, key):
            self.url = url
            self.key = key
            
        def table(self, table_name):
            class TableClient:
                def __init__(self, url, key, table):
                    self.url = url
                    self.key = key
                    self.table = table
                
                def select(self, cols="*"):
                    return self
                
                def eq(self, col, val):
                    return self
                
                def execute(self):
                    resp = requests.get(
                        f"{self.url}/rest/v1/{self.table}",
                        headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"},
                        timeout=10
                    )
                    class Result:
                        def __init__(self, data):
                            self.data = data
                    return Result(resp.json() if resp.status_code == 200 else [])
            return TableClient(url, key, table_name)
    return SimpleClient(SUPABASE_URL, SUPABASE_ANON_KEY)