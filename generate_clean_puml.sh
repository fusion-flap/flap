OUTPUT_DIR="docs/static"
echo "Using: $(which pyreverse)"
echo "Generating .puml files into '$OUTPUT_DIR/'..."
pyreverse -o puml -d $OUTPUT_DIR flap/
echo "Cleaning NoneType annotations..."
sed -i 's/ : NoneType$//g' "$OUTPUT_DIR/classes.puml"
sed -i 's/, NoneType$//g' "$OUTPUT_DIR/classes.puml"
sed -i 's/NoneType, //g' "$OUTPUT_DIR/classes.puml"

echo "Done."
echo "Generated PUML files at '$OUTPUT_DIR/classes.puml' and '$OUTPUT_DIR/packages.puml'."
echo "Use a PlantUML instance, (e.g. at https://www.plantuml.com/plantuml/) to generate diagrams in .svg or other formats."
