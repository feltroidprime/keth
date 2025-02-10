from typing import Optional

from ethereum.cancun.fork_types import EMPTY_ACCOUNT, Account
from ethereum_types.numeric import U256, Uint
from hypothesis import example, given


class TestForkTypes:
    def test_account_default(self, cairo_run):
        assert EMPTY_ACCOUNT == cairo_run("EMPTY_ACCOUNT")

    @given(account_a=..., account_b=...)
    @example(
        account_a=Account(nonce=Uint(1), balance=U256(2**128), code=bytearray()),
        account_b=Account(nonce=Uint(1), balance=U256(2**129), code=bytearray()),
    )
    @example(
        account_a=EMPTY_ACCOUNT,
        account_b=EMPTY_ACCOUNT,
    )
    def test_account_eq(
        self, cairo_run, account_a: Optional[Account], account_b: Optional[Account]
    ):
        assert (account_a == account_b) == cairo_run(
            "Account__eq__", account_a, account_b
        )
