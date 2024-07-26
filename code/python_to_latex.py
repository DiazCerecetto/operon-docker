from SRutils import python_to_latex, simplify_expression
expression_str = "((-0.000) + (1.000 * (((cos(1.158) - 0.375) + (((((-29.978) * X2) * ((-0.278) * X1)) + (cos((-0.876)) - (((-0.076) * X2) * ((-0.190) * X1)))) / ((4.361 * X3) / (0.002 - (0.177 * X3))))) * ((((-1.538) - (((-19.487) * X2) / ((0.144 * X2) - 3.712))) / (((-18.269) - ((((-2835.066) * X3) - 39249.930) / ((1184.482 * X2) + 1448.616))) - (((0.500 * X3) * (1.527 * X3)) / ((-0.186) * X2)))) + (-2.955)))))"
python_to_latex(simplify_expression(expression_str))
# Save image to file
