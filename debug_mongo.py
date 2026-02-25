"""
MongoDB Debug Script - tries multiple connection approaches
"""
from pymongo import MongoClient
import json

# Try different URI variations
URIS_TO_TRY = [
    "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster",
    "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/?authSource=admin",
    "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/",
    "mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/test",
]

SKIP_DBS = {"admin", "local", "config"}

def try_connect(uri, label):
    print(f"\n{'='*60}")
    print(f"Trying: {label}")
    print(f"URI: {uri[:60]}...")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000,
                             tlsAllowInvalidCertificates=True)
        client.admin.command("ping")
        print("✅ Connected!")
        return client
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def explore(client):
    print("\n" + "="*60)
    print("EXPLORING ALL DATABASES:")
    print("="*60)

    # Try listing databases
    try:
        dbs = client.list_database_names()
        print(f"Databases found: {dbs}")
    except Exception as e:
        print(f"Cannot list databases: {e}")
        dbs = []

    # Always try these DB names regardless
    force_try = ["dev-cluster", "dev_cluster", "devcluster", "test",
                 "admin", "mydb", "data", "app", "production", "dev"]
    all_names = list(set(list(dbs) + force_try))

    for db_name in all_names:
        try:
            db = client[db_name]
            cols = db.list_collection_names()
            if cols:
                print(f"\n✅ DB '{db_name}' has {len(cols)} collection(s): {cols}")
                for col in cols:
                    try:
                        count = db[col].count_documents({})
                        sample = list(db[col].find({}, {"_id": 0}).limit(1))
                        fields = list(sample[0].keys()) if sample else []
                        print(f"   📂 {col}: {count} docs | fields: {fields}")
                        if sample:
                            print(f"      Sample: {json.dumps(sample[0], default=str)}")
                    except Exception as e:
                        print(f"   📂 {col}: error - {e}")
        except Exception as e:
            pass  # silently skip

connected_client = None
for i, uri in enumerate(URIS_TO_TRY):
    c = try_connect(uri, f"URI variant {i+1}")
    if c:
        connected_client = c
        break

if connected_client:
    explore(connected_client)
else:
    print("\n" + "="*60)
    print("ALL CONNECTION ATTEMPTS FAILED")
    print("="*60)
    print("\nThis means your IP is NOT whitelisted in MongoDB Atlas.")
    print("\nTo fix:")
    print("1. Go to https://cloud.mongodb.com")
    print("2. Sign in → select your project")
    print("3. Left sidebar → Network Access")
    print("4. Click '+ ADD IP ADDRESS'")
    print("5. Click 'ALLOW ACCESS FROM ANYWHERE' (0.0.0.0/0)")
    print("6. Click Confirm, wait 30 seconds")
    print("7. Run this script again")