from typing import Dict
from model.model import NLPModel
from openfabric_pysdk.utility import SchemaUtil

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

# Load the Model
model = NLPModel()
model.load_model('model/model.h5')


############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    """
    Callback function called on each execution pass.

    :param request: Input request.
    :param ray: Openfabric execution ray.
    :param state: Openfabric ray state.
    :return: SimpleText response.
    """
    output = []
    for text in request.text:
        response = model.predict(text)
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))
