import pytest

from hope.data.generators import FAMILIES, family_for, generate
from hope.tools.builtin import default_catalog
from hope.tools.verify import verify_episode
from hope.constraints.critic import critique


@pytest.mark.parametrize("family", list(FAMILIES))
def test_generated_episodes_pass_independent_verification(family):
    cat = default_catalog()
    eps = generate(family, 6, seed=1, level=2)
    statuses = [verify_episode(cat, e).verification["status"] for e in eps]
    assert "REFUTADO" not in statuses, statuses
    assert statuses.count("RESPALDADO") >= 4 or family == "datos_faltantes"
    assert all("Resultado:" in e.training_text() for e in eps)


def test_generation_is_deterministic_and_diverse():
    a = [e.data["problema"] for e in generate("ohm", 20, seed=3)]
    b = [e.data["problema"] for e in generate("ohm", 20, seed=3)]
    assert a == b and len(set(a)) > 10


def test_topic_mapping_covers_curriculum():
    import yaml
    topics = [t["name"] for t in yaml.safe_load(open("configs/curriculum.yaml"))["topics"]]
    covered = [t for t in topics if family_for(t)]
    assert len(covered) >= 10


def test_missing_data_episodes_agree_with_the_critic():
    for e in generate("datos_faltantes", 6, seed=0):
        c = critique(e.data["problema"], e.training_text())
        assert c.verdict == "ok", (e.data["problema"], [i.detail for i in c.issues])
