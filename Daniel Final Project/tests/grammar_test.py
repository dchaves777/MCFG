import pytest
from MCFGGrammar import *

# Example Grammar Fixture
@pytest.fixture
def mcfg_grammar():
    rule_string = """S(uv) -> NP(u) VP(v)
S(uv) -> NPwh(u) VP(v)
S(vuw) -> Aux(u) Swhmain(v, w)
S(uwv) -> NPdisloc(u, v) VP(w)
S(uwv) -> NPwhdisloc(u, v) VP(w)
Sbar(uv) -> C(u) S(v)
Sbarwh(v, uw) -> C(u) Swhemb(v, w)
Sbarwh(u, v) -> NPwh(u) VP(v)
Swhmain(v, uw) -> NP(u) VPwhmain(v, w)
Swhmain(w, uxv) -> NPdisloc(u, v) VPwhmain(w, x)
Swhemb(v, uw) -> NP(u) VPwhemb(v, w)
Swhemb(w, uxv) -> NPdisloc(u, v) VPwhemb(w, x)
Src(v, uw) -> NP(u) VPrc(v, w)
Src(w, uxv) -> NPdisloc(u, v) VPrc(w, x)
Src(u, v) -> N(u) VP(v)
Swhrc(u, v) -> Nwh(u) VP(v)
Swhrc(v, uw) -> NP(u) VPwhrc(v, w)
Sbarwhrc(v, uw) -> C(u) Swhrc(v, w)
VP(uv) -> Vpres(u) NP(v)
VP(uv) -> Vpres(u) Sbar(v)
VPwhmain(u, v) -> NPwh(u) Vroot(v)
VPwhmain(u, wv) -> NPwhdisloc(u, v) Vroot(w)
VPwhmain(v, uw) -> Vroot(u) Sbarwh(v, w)
VPwhemb(u, v) -> NPwh(u) Vpres(v)
VPwhemb(u, wv) -> NPwhdisloc(u, v) Vpres(w)
VPwhemb(v, uw) -> Vpres(u) Sbarwh(v, w)
VPrc(u, v) -> N(u) Vpres(v)
VPrc(v, uw) -> Vpres(u) Nrc(v, w)
VPwhrc(u, v) -> Nwh(u) Vpres(v)
VPwhrc(v, uw) -> Vpres(u) Sbarwhrc(v, w)
NP(uv) -> D(u) N(v)
NP(uvw) -> D(u) Nrc(v, w)
NPdisloc(uv, w) -> D(u) Nrc(v, w)
NPwh(uv) -> Dwh(u) N(v)
NPwh(uvw) -> Dwh(u) Nrc(v, w)
NPwhdisloc(uv, w) -> Dwh(u) Nrc(v, w)
Nrc(v, uw) -> C(u) Src(v, w)
Nrc(u, vw) -> N(u) Swhrc(v, w)
Nrc(u, vwx) -> Nrc(u, v) Swhrc(w, x)
Dwh(which)
Nwh(who)
D(the)
D(a)
N(greyhound)
N(human)
Vpres(believes)
Vroot(believe)
Aux(does)
C(that)"""
    grammar =  MCFGGrammar.from_string(rule_string)
    return MCFGParser(grammar)


# Test MCFG Parser
@pytest.fixture
def test_mcfg_parser(mcfg_grammar):
    assert mcfg_grammar(["which", "greyhound", "does", "the", "human", "believe"], mode = "recognize") == True
    assert mcfg_grammar(["which", "human", "believes", "a", "greyhound"], mode="recognize") == True
    assert mcfg_grammar(["does", "the", "greyhound", "believe", "the", "human"], mode = "recognize") == False
    assert mcfg_grammar(["the", "greyhound", "runs"], mode="recognize") == False
    print('All parser tests in mode "recognize" have passed!')

@pytest.fixture
def test_mcfg_tree(mcfg_grammar):
    assert list(mcfg_grammar(["which", "greyhound", "does", "the", "human", "believe"], mode = "parse"))[0].to_tuple()[0].variable == 'S'
    assert list(mcfg_grammar(["which", "human", "believes", "a", "greyhound"], mode="parse"))[0].to_tuple()[0].variable == 'S'
    assert mcfg_grammar(["does", "the", "greyhound", "believe", "the", "human"], mode = "parse") == set()
    assert mcfg_grammar(["the", "greyhound", "runs"], mode="parse") == set()
    print('All parser tests in mode "parse" have passed!')
