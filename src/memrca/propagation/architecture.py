"""Documented call dependencies of the benchmark applications, for drawing only.

The trace graph misses calls that were never traced: Sock Shop has no traces in RCAEval,
and databases and caches rarely appear in spans. The console draws these documented
dependencies as dashed links so that every service has a place in the dependency graph.
PropagationNet and all experiments use the traced graph only.

Sources: the architecture diagrams of Online Boutique (GoogleCloudPlatform/microservices-demo)
and Sock Shop (microservices-demo/microservices-demo); for Train Ticket, each ``ts-X-mongo``
or ``ts-X-mysql`` database belongs to ``ts-X-service``.
"""
from __future__ import annotations

ONLINE_BOUTIQUE = [
    ("frontend-external", "frontend"),
    *[("frontend", s) for s in ("adservice", "cartservice", "checkoutservice", "currencyservice",
                                "productcatalogservice", "recommendationservice", "shippingservice")],
    *[("checkoutservice", s) for s in ("cartservice", "currencyservice", "emailservice", "paymentservice",
                                       "productcatalogservice", "shippingservice")],
    ("recommendationservice", "productcatalogservice"),
    ("cartservice", "redis"),
]

SOCK_SHOP = [
    *[("front-end", s) for s in ("catalogue", "carts", "orders", "user", "session-db")],
    *[("orders", s) for s in ("carts", "user", "payment", "shipping", "orders-db")],
    ("catalogue", "catalogue-db"), ("carts", "carts-db"), ("user", "user-db"),
    ("shipping", "rabbitmq"), ("queue-master", "rabbitmq"), ("rabbitmq-exporter", "rabbitmq"),
]


def documented_edges(system: str, services: list[str]) -> list[tuple[str, str]]:
    """Caller -> callee pairs among ``services`` for ``system`` ('ob', 'ss' or 'tt')."""
    present = set(services)
    if system == "ob":
        pairs = ONLINE_BOUTIQUE
    elif system == "ss":
        pairs = SOCK_SHOP
    elif system == "tt":
        pairs = []
        for s in services:
            for suffix in ("-mongo", "-mysql"):
                if s.endswith(suffix):
                    pairs.append((s[: -len(suffix)] + "-service", s))
    else:
        pairs = []
    return [(a, b) for a, b in pairs if a in present and b in present]
