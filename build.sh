mkdir tmp 
mkdir output

# Simulator 
cd simulator
g++ -O3 simulator.cpp -o simulator
g++ -O3 wl_simulator.cpp -o wl_simulator

# Analyzer 
cd ../cone_analyzer 
g++ -O3 analyzer.cpp -o analyzer