from hope.teacher.cycle import answers_match, extract_answer
from hope.train.star import StarConfig, self_improve

P = [{"tema": "ley de ohm", "problema": "¿Qué tensión cae en un resistor de 220 ohm con 0.05 A?", "respuesta": "11 V"}]


def test_star_keeps_only_verified_solutions():
    outs = iter(["Paso 1: 220 * 0,05 = 12\nResultado: 11 V",       # acierta pero el paso está mal → «suerte»
                 "Paso 1: 220 * 0,05 = 11\nResultado: 11 V"])
    rep = self_improve(lambda p, t: next(outs), P, extract=extract_answer, match=answers_match,
                       cfg=StarConfig(samples=2))
    assert rep.solved_directly == 1 and rep.rejected_lucky == 1
    assert rep.texts[0].startswith("Tema: ley de ohm\nProblema:") and "= 11" in rep.texts[0]


def test_star_rationalizes_without_keeping_the_hint():
    seen = []

    def gen(prompt, temp):
        seen.append(prompt)
        return "Paso 1: 220 * 0,05 = 11\nResultado: 11 V" if "Pista" in prompt else "Resultado: 3 V"
    rep = self_improve(gen, P, extract=extract_answer, match=answers_match, cfg=StarConfig(samples=2))
    assert rep.rationalized == 1 and "Pista" in seen[-1]
    assert "Pista" not in rep.texts[0]


def test_star_rejects_everything_wrong():
    rep = self_improve(lambda p, t: "Resultado: 3 V", P, extract=extract_answer, match=answers_match)
    assert rep.texts == [] and rep.problems == 1
