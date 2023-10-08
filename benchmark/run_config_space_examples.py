from configuration_space import Group_Configuration_Space


Test_Object = Group_Configuration_Space()


print(Test_Object.get_mango_config_space())
print(Test_Object.get_smac_configuration_space())
print(Test_Object.get_hyperopt_configspace()['algorithm'])
print(Test_Object.get_optuna_space())