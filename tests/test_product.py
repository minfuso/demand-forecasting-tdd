import pytest
from dataclasses import FrozenInstanceError
from src.domain.product import Product

def test_product_can_be_instancied():
    product = Product(product_id="A")
    assert product.product_id == "A"
    
def test_product_is_immutable():
    p = Product(product_id="A")
    with pytest.raises(FrozenInstanceError):
        p.product_id = "B"