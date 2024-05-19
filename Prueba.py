from model import model
from pgmpy.inference import VariableElimination

# Crear una instancia de VariableElimination
infer = VariableElimination(model)

prob_total = infer.query(variables=['MotherGen',"MotherTrait", "FatherGen","FatherTrait","ChildrenGen", "ChildrenTrait"])
print(prob_total)

hola= model.get_state_probability({"MotherGen": "0gen", "MotherTrait": "no", "FatherGen": "2gen", "FatherTrait": "yes", "ChildrenGen": "1gen", "ChildrenTrait":"no"})
print(hola)


predictions = infer.query(variables=['MotherGen',"FatherGen","ChildrenGen", "ChildrenTrait"], evidence={"FatherTrait":"yes", "MotherTrait":"no"}, joint=False)


for factor in predictions.values():
    print(factor)