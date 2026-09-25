#  Abrir Terminal Ctrl+Shift+P -> “Julia: Start REPL”

# Instalar Paquetes adicionales
begin
    import Pkg;

    # Lista de paquetes a instalar en bloque (mucho más rápido que uno a uno)
    paquetes = [
        "FileIO",
        "XLSX",
        "JLD2",
        "Flux",
        "Plots",
        "MAT",
        "Images",
        "CSV",
        "DelimitedFiles",
        "MLJ",
        "LIBSVM",
        "MLJLIBSVMInterface",
        "NearestNeighborModels",
        "MLJDecisionTreeInterface",
        "StatsBase",
    ];
    Pkg.add(paquetes);

    # --------------Paquetes con versiones especificas-----------------
    # Fijamos SymDoME exactamente en la versión 1.0.4.
    # Usamos Pkg.pin para anclarlo: esto garantiza que conserve esa versión
    # y evita que futuras actualizaciones o Pkg.update() lo cambien.
    symdome_spec = Pkg.PackageSpec(name="SymDoME", version="1.0.4");
    Pkg.add(symdome_spec);
    Pkg.pin(symdome_spec);
    #------------------------------------------------------------------
end

begin
    #Equivalent to python would be:
    #from Statistics import *
    using Statistics

    using FileIO
    using Flux
    using Flux.Losses
    using DelimitedFiles

    using Random
    using Random:seed!

    using SymDoME

    using MLJ
    using LIBSVM
    using MLJLIBSVMInterface
    using NearestNeighborModels
    using MLJDecisionTreeInterface
end

# Read Iris Dataset (Comprobación de lectura)
begin
    # usamos @__DIR__ para ruta absoluta
    # readdlm(archivo, delimitador) -> ',' es el delimitador, le dice
    # que caracter se usa para separar elementos, en este caso la coma:
    # ','
    dataset = readdlm(joinpath(@__DIR__, "iris.data"), ',');
    inputs = dataset[:,1:4];
    targets = dataset[:,5];

    println("Tipo dataset: ", typeof(dataset))
    println("Tipo inputs:  ", typeof(inputs))
    println("Tipo targets: ", typeof(targets))

    println("¿Mismo tipo inputs y targets?: ", typeof(inputs) isa typeof(targets))
end

####################################################
# En caso de errores
####################################################

# Info General
# 1. La carpeta .julia persiste entre instalaciones, hay que eliminar la carpeta para poder tener una instalación limpia
# 2. La carpeta .julia/conda contiene el entorno miniconda de python que julia usa para PyCall (se genera en ejecución)
# 3. Borrar la carpeta .julia/registries/General y ejecutando Pkg.update() puede solucionar problemas con los paquetes
# 4. La version de python puede cambiarse con los comandos: [ENV["PYTHON"]="C:/path/python.exe" + Pkg.build("PyCall") + reiniciar Julia]
#    Los cambios en la versión de python pueden revertirse con: [ENV["PYTHON"]="" + Pkg.build("PyCall") + reiniciar Julia]
# 5. Puede usarse un conda preexistente con  [run(`conda create -n conda_jl python conda`) + ENV["CONDA_JL_HOME"] = "/path/conda_jl"  +  Pkg.build("Conda") + reiniciar Julia]


# Error con ScikitLearn (relacionado con el paquete PyCall)
####################################################

# Solucion 1
#############
# Construir PyCall para nuestro Python y reintentar
# 1) Ejecutar el siguiente bloque
# begin
#     import Pkg;
#     Pkg.update();
#     Pkg.build("PyCall");
# end
# 2) Cerrar RPL Julia y reintentar los pasos anteriores

####################################################
# Error Fallo de instalacion en Python de SKlearn
####################################################

# Solucion 1
#############
# Instalar ScikitLearn en nuestro entorno python de Julia a mano desde el conda de Julia
# 1.) CMD:> cd C:\Users\o_siy\.julia\conda\3\Scripts
# 2.) CMD:> conda env list
# 3.) CMD:> conda install -p C:\Users\o_siy\.julia\conda\3 scikit-learn
# 4.) Cerrar Julia, Reiniciar y reintentar

# Solucion 2
#############
# Instalar ScikitLearn en nuestro entorno python de Julia a mano desde pip
# En Windows
# 1.) CMD:> cd C:\Users\o_siy\.julia\conda\3
# 2.) CMD:> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# 3.) CMD:> python get-pip.py
# 4.) CMD:> pip install -U scikit-learn
# En Linux
# 1.) py get-pip.py
# 2.) pip install -U scikit-learn

####################################################



####################################################
# Error de Version de Python
####################################################

# Solucion 1
#############
# Version de Python en Julia Incorrecta
# 1.) Cambiar version de Python para PyCall
# ENV["PYTHON"]="C:/Users/o_siy/.julia/conda/3/python.exe" # Ejemplo
# 2.) Reconstruir Pycall
# Pkg.build("PyCall");
# 3.) Comprobar version de PyCall construida en el log del terminal Ej: (Building PyCall → `C:\Users\o_siy\.julia\scratchspaces\44cfe95a-1eb2-52ea-b672-e2afdf69b78f\9816a3826b0ebf49ab4926e2b18842ad8b5c8f04\build.log`)
# 4.) Relanzar RPL
# 5.) Reintentar

# Solucion 2
#############
# La version de Julia de Python es incompatible con Sklearn y tenemos una version adecuada de python en el systema
# 1.) Cambiar version de Python para PyCall
# ENV["PYTHON"]="C:/ProgramData/anaconda3/python.exe"  # Ejemplo
# 2.) Reconstruir Pycall
# import Pkg;
# Pkg.build("PyCall");
# 3.) Comprobar version de PyCall construida en el log del terminal Ej: (Building PyCall → `C:\Users\o_siy\.julia\scratchspaces\44cfe95a-1eb2-52ea-b672-e2afdf69b78f\9816a3826b0ebf49ab4926e2b18842ad8b5c8f04\build.log`)
# 4.) Relanzar RPL
# 5.) Reintentar

# Solucion 3
#############
# La version de Julia de Python es incompatible con Sklearn y no tenemos una version adecuada de python en el systema
# 1.) Descargar una nueva version de python
# https://www.python.org/downloads/
# 2.) Instalar Python
# 3.) Crear Entorno de anaconda desde el conda de Julia
# CMD:> cd C:\Users\o_siy\.julia\conda\3\Scripts
# conda create --prefix "C:/Python312" python=3.12
# 4.) Cambiar version de Python para PyCall
# ENV["PYTHON"]="C:/Python312/python.exe"  # Ejemplo
# 5.) Reconstruir Pycall
# import Pkg;
# Pkg.build("PyCall");
# 6.) Comprobar version de PyCall construida en el log del terminal Ej: (Building PyCall → `C:\Users\o_siy\.julia\scratchspaces\44cfe95a-1eb2-52ea-b672-e2afdf69b78f\9816a3826b0ebf49ab4926e2b18842ad8b5c8f04\build.log`)
# 7.) Relanzar RPL
# 8.) Reintentar

####################################################


####################################################
# Otros Errores
####################################################

# https://discourse.julialang.org/t/a-guide-how-to-handle-error-unsatisfiable-requirements-detected/43406
# https://discourse.julialang.org/t/error-unsatisfiable-requirements-detected-for-package/54901/4
# https://stackoverflow.com/questions/64359283/why-am-i-getting-unsatisfiable-requirements-detected-for-package-http-error-wh
# https://stackoverflow.com/questions/57639110/julia-how-to-update-to-the-latest-version-of-a-package-i-e-flux
# https://stackoverflow.com/questions/64846259/julia-package-has-no-known-versions-when-trying-to-add-a-package

