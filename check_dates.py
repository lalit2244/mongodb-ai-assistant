from pymongo import MongoClient
import json

client = MongoClient("mongodb+srv://mcpaccess:mcpaccess@dev6.4hksq.mongodb.net/dev-cluster", serverSelectionTimeoutMS=10000)
db = client["dev-cluster"]

# Check exact issueDate format and range in Voucher
print("=== VOUCHER DATE SAMPLES ===")
samples = list(db["Voucher"].find({"type": "sales"}, {"_id":0,"issueDate":1,"createdAt":1,"billFinalAmount":1}, limit=5))
for s in samples:
    print(s)

# Find min/max dates
print("\n=== DATE RANGE ===")
r = list(db["Voucher"].aggregate([
    {"$match": {"type": "sales", "iCompanyId": {"$ne": None}}},
    {"$group": {"_id": None, "minDate": {"$min": "$issueDate"}, "maxDate": {"$max": "$issueDate"}, "count": {"$sum": 1}}}
]))
print(r)

# Check what 2025 data looks like
print("\n=== 2025 SALES COUNT ===")
r2 = list(db["Voucher"].aggregate([
    {"$match": {"type": "sales", "issueDate": {"$gte": "2025-01-01"}}},
    {"$count": "total"}
]))
print(r2)

# Check 2024 data
print("\n=== 2024 SALES COUNT ===")
r3 = list(db["Voucher"].aggregate([
    {"$match": {"type": "sales", "issueDate": {"$gte": "2024-01-01", "$lt": "2025-01-01"}}},
    {"$count": "total"}
]))
print(r3)

# Check issueDate type
print("\n=== FIELD TYPES ===")
sample = db["Voucher"].find_one({"type":"sales"}, {"_id":0,"issueDate":1,"createdAt":1})
print(f"issueDate type: {type(sample['issueDate'])}, value: {repr(sample['issueDate'])}")
print(f"createdAt type: {type(sample.get('createdAt'))}, value: {repr(sample.get('createdAt'))}")

# Check ItemQuantityTracker date range
print("\n=== ItemQuantityTracker YEAR RANGE ===")
r4 = list(db["ItemQuantityTracker"].aggregate([
    {"$group": {"_id": None, "minYear": {"$min": "$year"}, "maxYear": {"$max": "$year"}, 
                "minMonth": {"$min": "$month"}, "maxMonth": {"$max": "$month"}}}
]))
print(r4)

# Sales last month via ItemQuantityTracker
print("\n=== ItemQuantityTracker Jan 2025 SALES ===")
r5 = list(db["ItemQuantityTracker"].aggregate([
    {"$match": {"voucherType": "sales", "year": 2025, "month": 1}},
    {"$group": {"_id": None, "total_qty": {"$sum": "$qty"}, "total_amount": {"$sum": "$amount"}, "count": {"$sum": 1}}}
]))
print(r5)

print("\n=== ItemQuantityTracker Feb 2025 SALES ===")
r6 = list(db["ItemQuantityTracker"].aggregate([
    {"$match": {"voucherType": "sales", "year": 2025, "month": 2}},
    {"$group": {"_id": None, "total_qty": {"$sum": "$qty"}, "total_amount": {"$sum": "$amount"}, "count": {"$sum": 1}}}
]))
print(r6)