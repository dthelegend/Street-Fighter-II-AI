import retro
from street_fighter_ii_ai.wrappers import RetroOutputToTFTensorObservation, GrayScaleObservation, ResizeObservation, DiscretisedAction
from street_fighter_ii_ai import ROOT_DIR

CUSTOM_INTEGRATIONS_PATH = ROOT_DIR / "custom_integrations"

print("Loading custom integrations from " + str(CUSTOM_INTEGRATIONS_PATH) + "...")
retro.data.Integrations.add_custom_path(CUSTOM_INTEGRATIONS_PATH)

print("StreetFighterIISpecialChampionEdition-Genesis" in retro.data.list_games(inttype=retro.data.Integrations.ALL))

def create_environment(state = retro.State.DEFAULT):
    env = retro.make("StreetFighterIISpecialChampionEdition-Genesis", inttype=retro.data.Integrations.ALL, state = state)

    env = RetroOutputToTFTensorObservation(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = DiscretisedAction(env)

    return env
