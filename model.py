from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Definición del modelo
model = BayesianNetwork([('MotherGen', 'MotherTrait'), 
                       ('MotherGen', 'ChildrenGen'), 
                       ('FatherGen', 'FatherTrait'), 
                       ('FatherGen', 'ChildrenGen'),
                       ("ChildrenGen", "ChildrenTrait")])

# Definición de las distribuciones de probabilidad condicional
cpd_MotherGen = TabularCPD(variable='MotherGen', variable_card=3, 
                      values=[[0.01], [0.03], [0.96]], 
                      state_names={'MotherGen': ["2gen", "1gen", "0gen"]})

cpd_MotherTrait = TabularCPD(variable='MotherTrait', variable_card=2, 
                             evidence=['MotherGen'], evidence_card=[3],
                             values=[[0.65, 0.56, 0.01], 
                                     [0.35, 0.44, 0.99]],
                             state_names={'MotherGen': ["2gen", "1gen", "0gen"], 
                                          'MotherTrait': ["yes", "no"]})

cpd_FatherGen = TabularCPD(variable='FatherGen', variable_card=3, 
                      values=[[0.01], [0.03], [0.96]], 
                      state_names={'FatherGen': ["2gen", "1gen", "0gen"]})

cpd_FatherTrait = TabularCPD(variable='FatherTrait', variable_card=2, 
                             evidence=['FatherGen'], evidence_card=[3],
                             values=[[0.65, 0.56, 0.01], 
                                     [0.35, 0.44, 0.99]],
                             state_names={'FatherGen': ["2gen", "1gen", "0gen"], 
                                          'FatherTrait': ["yes", "no"]})


cpd_ChildrenGen = TabularCPD(variable='ChildrenGen', variable_card=3, 
                       evidence=['MotherGen', 'FatherGen'], evidence_card=[3, 3],
                       values=[[0.98, 0.495, 0.49, 0.495, 0.01, 0.01, 0.49, 0.01, 0.0], 
                               [0.01, 0.495, 0.49, 0.495, 0.98, 0.495, 0.49, 0.495, 0.01],
                               [0.01, 0.01, 0.02, 0.01, 0.01, 0.495, 0.02, 0.495, 0.99]],
                       state_names={'MotherGen': ["2gen", "1gen", "0gen"], 
                                    'FatherGen': ["2gen", "1gen", "0gen"], 
                                    'ChildrenGen': ["2gen", "1gen", "0gen"]})

cpd_ChildrenTrait = TabularCPD(variable='ChildrenTrait', variable_card=2, 
                             evidence=['ChildrenGen'], evidence_card=[3],
                             values=[[0.65, 0.56, 0.01], 
                                     [0.35, 0.44, 0.99]],
                             state_names={'ChildrenGen': ["2gen", "1gen", "0gen"], 
                                          'ChildrenTrait': ["yes", "no"]})

# Añadir las distribuciones al modelo
model.add_cpds(cpd_MotherGen, cpd_MotherTrait, cpd_FatherGen, cpd_FatherTrait, cpd_ChildrenGen, cpd_ChildrenTrait)


# Verificar el modelo: devuelve True si el modelo es correcto
model.check_model()