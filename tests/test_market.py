"""Check quote import, market IV, and Alpaca's paginated snapshot format."""

from datetime import date
from unittest import TestCase
from unittest.mock import patch

from options_lab import alpaca_data, market, pricing
from options_lab.server import compute


class MarketChainTests(TestCase):
    def test_sample_csv_round_trip(self):
        contents = market.example_csv()
        self.assertEqual(market.read_csv(contents), market.example_quotes())
        rows = market.analyze(market.read_csv(contents), [("Black–Scholes", pricing)],
                              as_of=market.EXAMPLE_DATE, spot=100, rate=.04,
                              dividend=0, reference_vol=.25)
        self.assertEqual(len(rows), 54)
        self.assertTrue(all(row["iv_status"] == "recovered" for row in rows))
        atm = next(row for row in rows if row["days"] == 36 and row["strike"] == 100 and row["kind"] == "call")
        self.assertAlmostEqual(atm["iv_pct"], 25, delta=.02)
        self.assertAlmostEqual(atm["model_prices"]["Black–Scholes"], 3.3260841169)

    def test_model_uses_reference_iv_and_marks_bad_or_old_quotes(self):
        csv = ("expiry,strike,kind,bid,ask,quote_time\n"
               "2026-02-20,90,call,12,11,\n"
               "2026-02-20,95,call,0,1,\n"
               "2026-02-20,100,call,3.24,3.41,2026-01-14T15:55:00-05:00\n"
               "2026-02-20,105,call,1,2,2026-01-16T20:55:00Z\n"
               "2026-02-20,110,call,,1,\n"
               "2026-02-20,120,call,100,102,\n"
               "2026-01-15,125,call,1,2,\n")
        rows = {row["strike"]: row for row in market.analyze(
            market.read_csv(csv), [("Black–Scholes", pricing)], as_of=market.EXAMPLE_DATE,
            spot=100, rate=.04, dividend=0, reference_vol=.50)}
        self.assertEqual(rows[90]["quote_status"], "crossed")
        self.assertEqual(rows[95]["quote_status"], "zero_side")
        self.assertIsNone(rows[95]["mid"])
        self.assertEqual(rows[100]["quote_age_days"], 1)
        self.assertAlmostEqual(rows[100]["mid"], 3.325)
        self.assertAlmostEqual(rows[100]["iv_pct"], 25, delta=.02)
        self.assertAlmostEqual(rows[100]["model_prices"]["Black–Scholes"],
                               pricing.price(100, 100, 36 / 365, .50, .04))
        self.assertEqual(rows[105]["iv_status"], "after_valuation_date")
        self.assertIsNone(rows[105]["iv_pct"])
        self.assertEqual(rows[110]["quote_status"], "missing_side")
        self.assertEqual(rows[120]["iv_status"], "outside_european_bounds")
        self.assertEqual(rows[125]["iv_status"], "unsupported_expiry")
        self.assertEqual(rows[125]["model_prices"], {})

    def test_rejects_malformed_csv_or_inputs_before_model_training(self):
        header = "expiry,strike,kind,bid,ask\n"
        quote = "2026-02-20,100,call,2,3\n"
        for contents in (header + quote + quote, header + quote.replace(",2,", ",nan,"),
                         header + quote[:-1] + ",extra\n", "wrong,headers\n", header):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                market.read_csv(contents)
        with self.assertRaisesRegex(ValueError, "Valuation date"):
            compute("/api/market", {"source": "csv", "as_of": "2026-02-30", "quotes_csv": header + quote})
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            market.timestamp("2026-01-15T20:55:00+03:76")

    @patch.dict("os.environ", {"APCA_API_KEY_ID": "fixture-key", "APCA_API_SECRET_KEY": "fixture-secret"})
    @patch("options_lab.alpaca_data._get")
    def test_alpaca_snapshot_reads_spot_and_pages_skipping_adjusted_roots(self, get):
        quote = {"latestQuote": {"bp": 2.45, "ap": 2.55, "t": "2026-01-15T20:55:00.123456789Z"}}
        get.side_effect = [
            {"trade": {"p": 100, "t": "2026-01-15T20:55:01Z"}},
            {"snapshots": {"AAPL260220C00100000": quote, "AAPL1260220C00105000": quote},
             "next_page_token": "fixture/page"},
            {"snapshots": {"AAPL260220P00102500": quote}, "next_page_token": None},
        ]
        expiry = date(2026, 2, 20)
        snapshot = alpaca_data.snapshot("AAPL", "indicative", expiry, market.EXAMPLE_DATE)
        self.assertEqual(snapshot["spot"], 100)
        self.assertFalse(snapshot["truncated"])
        self.assertEqual([(q.kind, q.strike) for q in snapshot["quotes"]], [("call", 100), ("put", 102.5)])
        self.assertEqual(snapshot["quotes"][0].quote_time.microsecond, 123456)
        self.assertEqual(get.call_args_list[0].args[:2], ("/v2/stocks/AAPL/trades/latest", {"feed": "iex"}))
        params = get.call_args.args[1]
        self.assertEqual((params["expiration_date"], params["page_token"]), ("2026-02-20", "fixture/page"))
        self.assertEqual(get.call_args.args[2]["APCA-API-KEY-ID"], "fixture-key")
