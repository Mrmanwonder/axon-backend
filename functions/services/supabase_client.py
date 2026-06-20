import httpx

SUPABASE_URL = "https://anmfwzxyvqxyxxeobxti.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_fIbfGtT5yyFaogq4DQAuxw_tZ54kolM"


def get_supabase_client():
    """Simple Supabase client using httpx."""

    class SimpleClient:
        def __init__(self, url, key):
            self.url = url
            self.key = key
            self.client = httpx.Client(timeout=15)

        def table(self, table_name):
            return _TableClient(self.url, self.key, table_name, self.client)

        def close(self):
            self.client.close()

    class _TableClient:
        def __init__(self, url, key, table, client):
            self.url = url
            self.key = key
            self.table = table
            self.client = client
            self._filters = {}

        def select(self, cols="*"):
            return self

        def eq(self, col, val):
            self._filters[col] = val
            return self

        def execute(self):
            params = self._filters.copy()
            resp = self.client.get(
                f"{self.url}/rest/v1/{self.table}",
                headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"},
                params=params,
            )
            return _Result(resp.json() if resp.status_code == 200 else [])

    class _Result:
        def __init__(self, data):
            self.data = data

    return SimpleClient(SUPABASE_URL, SUPABASE_ANON_KEY)
