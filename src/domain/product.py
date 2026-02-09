from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    product_id: str
