import json
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("ecommerce_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Product Catalog (Vishal Shop)
# -------------------------
CATALOG = [
    {"id": "mug-001", "name": "Stoneware Chai Mug", "description": "Hand-glazed ceramic mug perfect for masala chai.", "price": 299, "currency": "INR", "category": "mug", "color": "blue", "sizes": []},
    {"id": "tee-001", "name": "Vishal Shop Tee (Cotton)", "description": "Comfort-fit cotton t-shirt with subtle logo.", "price": 799, "currency": "INR", "category": "tshirt", "color": "black", "sizes": ["S", "M", "L", "XL"]},
    {"id": "hoodie-001", "name": "Cozy Hoodie", "description": "Warm pullover hoodie, fleece-lined.", "price": 1499, "currency": "INR", "category": "hoodie", "color": "grey", "sizes": ["M", "L", "XL"]},
    {"id": "mug-002", "name": "Insulated Travel Mug", "description": "Keeps chai warm on your way to work.", "price": 599, "currency": "INR", "category": "mug", "color": "white", "sizes": []},
    {"id": "hoodie-002", "name": "Black Zip Hoodie", "description": "Lightweight zip-up hoodie, black.", "price": 1299, "currency": "INR", "category": "hoodie", "color": "black", "sizes": ["S", "M", "L"]},
    # Additional products...
]

ORDERS_FILE = "orders.json"
if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w") as f:
        json.dump([], f)

# -------------------------
# User Session Data
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    cart: List[Dict] = field(default_factory=list)
    orders: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)

# -------------------------
# Helper Functions
# -------------------------
def _load_all_orders() -> List[Dict]:
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _save_order(order: Dict):
    orders = _load_all_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

def list_products(filters: Optional[Dict] = None) -> List[Dict]:
    filters = filters or {}
    results = []
    query = filters.get("q")
    category = filters.get("category")
    max_price = filters.get("max_price") or filters.get("to") or filters.get("max")
    min_price = filters.get("min_price") or filters.get("from") or filters.get("min")
    color = filters.get("color")
    size = filters.get("size")
    if category:
        cat = category.lower()
        if cat in ("phone", "phones", "mobile", "mobile phone", "mobiles"):
            category = "mobile"
        elif cat in ("tshirt", "t-shirts", "tees", "tee"):
            category = "tshirt"
        else:
            category = cat
    for p in CATALOG:
        ok = True
        if category:
            pcat = p.get("category", "").lower()
            if pcat != category and category not in pcat and pcat not in category:
                ok = False
        if max_price:
            try: 
                if p.get("price", 0) > int(max_price): ok = False
            except Exception: pass
        if min_price:
            try: 
                if p.get("price", 0) < int(min_price): ok = False
            except Exception: pass
        if color and p.get("color") != color: ok = False
        if size and (not p.get("sizes") or size not in p.get("sizes")): ok = False
        if query:
            q = query.lower()
            if "phone" in q or "mobile" in q:
                if p.get("category") != "mobile": ok = False
            else:
                if q not in p.get("name", "").lower() and q not in p.get("description", "").lower():
                    ok = False
        if ok: results.append(p)
    return results

def find_product_by_ref(ref_text: str, candidates: Optional[List[Dict]] = None) -> Optional[Dict]:
    ref = (ref_text or "").lower().strip()
    cand = candidates if candidates is not None else CATALOG
    wants_mobile = any(w in ref for w in ("phone", "phones", "mobile", "mobiles"))
    filtered = [p for p in cand if p.get("category") == "mobile"] if wants_mobile else cand
    ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3}
    for word, idx in ordinals.items():
        if word in ref and idx < len(filtered):
            return filtered[idx]
    for p in cand:
        if p["id"].lower() == ref: return p
    for p in filtered:
        name = p["name"].lower()
        if all(tok in name for tok in ref.split() if len(tok) > 2): return p
    for p in cand:
        for tok in ref.split():
            if len(tok) > 2 and tok in p["name"].lower(): return p
    for token in ref.split():
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(filtered): return filtered[idx]
    for word, idx in ordinals.items():
        if word in ref and idx < len(cand): return cand[idx]
    return None

# -------------------------
# Tool Functions
# -------------------------
@function_tool
async def show_catalog(
    ctx: RunContext[Userdata], 
    q: Annotated[Optional[str], Field(default=None)] = None, 
    category: Annotated[Optional[str], Field(default=None)] = None,
    max_price: Annotated[Optional[int], Field(default=None)] = None, 
    color: Annotated[Optional[str], Field(default=None)] = None
) -> str:
    userdata = ctx.userdata
    filters = {"q": q, "category": category, "max_price": max_price, "color": color}
    prods = list_products({k: v for k, v in filters.items() if v is not None})
    if not prods:
        return "Sorry — I couldn't find any items that match. Would you like to try another search?"
    lines = [f"Here are the top {min(8, len(prods))} items I found at **Vishal Shop**:"]
    for idx, p in enumerate(prods[:8], start=1):
        size_info = f" (sizes: {', '.join(p['sizes'])})" if p.get('sizes') else ""
        lines.append(f"{idx}. {p['name']} — {p['price']} {p['currency']} (id: {p['id']}){size_info}")
    lines.append("You can say: 'I want the second item in size M' or 'add mug-001 to my cart, quantity 2'.")
    return "\n".join(lines)

@function_tool
async def add_to_cart(ctx: RunContext[Userdata], product_id: str, quantity: int = 1) -> str:
    userdata = ctx.userdata
    product = find_product_by_ref(product_id)
    if not product:
        return f"Sorry, I couldn't find a product with id '{product_id}'."
    for item in userdata.cart:
        if item["product_id"] == product["id"]:
            item["quantity"] += quantity
            break
    else:
        userdata.cart.append({"product_id": product["id"], "quantity": quantity})
    return f"Added {quantity} x {product['name']} to your cart."

@function_tool
async def show_cart(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty."
    lines = ["Here is your cart:"]
    for item in userdata.cart:
        prod = find_product_by_ref(item["product_id"])
        if prod:
            lines.append(f"{item['quantity']} x {prod['name']} — {prod['price']} {prod['currency']} each")
    return "\n".join(lines)

@function_tool
async def clear_cart(ctx: RunContext[Userdata]) -> str:
    ctx.userdata.cart.clear()
    return "Your cart has been cleared."

@function_tool
async def place_order(ctx: RunContext[Userdata]) -> str:
    if not ctx.userdata.cart:
        return "Your cart is empty, cannot place an order."
    order = {
        "order_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "items": ctx.userdata.cart.copy()
    }
    _save_order(order)
    ctx.userdata.orders.append(order)
    ctx.userdata.cart.clear()
    return f"Order placed successfully! Your order ID is {order['order_id']}."

@function_tool
async def last_order(ctx: RunContext[Userdata]) -> str:
    if not ctx.userdata.orders:
        return "You have not placed any orders yet."
    order = ctx.userdata.orders[-1]
    lines = [f"Your last order (ID: {order['order_id']}):"]
    for item in order["items"]:
        prod = find_product_by_ref(item["product_id"])
        if prod:
            lines.append(f"{item['quantity']} x {prod['name']} — {prod['price']} {prod['currency']}")
    return "\n".join(lines)

# -------------------------
# E-commerce Agent
# -------------------------
class ECommerceAgent(Agent):
    def __init__(self):
        instructions = """
        You are the **E-commerce Agent**, the official voice assistant for **Vishal Shop**.
        Universe: A friendly, efficient online Indian shop.
        Tone: Professional, highly helpful, and efficient; keep sentences short for TTS clarity.
        Role: Help the customer browse the catalog, add items to cart, place orders, and review recent orders.
        """
        super().__init__(
            instructions=instructions,
            tools=[show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order],
        )

# -------------------------
# Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🛍️" * 6)
    logger.info("🚀 STARTING VOICE E-COMMERCE AGENT (Vishal Shop)")

    userdata = Userdata()
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-marcus", style="Conversational", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )
    await session.start(
        agent=ECommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))