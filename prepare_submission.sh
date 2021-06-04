echo Preparing and zipping code
echo ' => Copying'
mkdir code
cp -r mvb         code/
cp -r NeurIPS2021 code/experiments
cp -r README_anon code/README
cp -r LICENSE     code/LICENSE_masegosa
echo ' => Removing binary files'
rm -rf code/mvb/bounds/__pycache__
rm -rf code/mvb/__pycache__
rm -rf code/experiments/__pycache__
echo ' => Checking for names in included code (There must be no input below!)'
echo [Start]
grep -nir "yi-shan" code
grep -nir wu code
grep -nir stephan code
grep -nir lorenzen code
grep -nir yevgeny code
grep -nir seldin code
grep -nir christian code
grep -nir igel code
grep -nir andres code
grep -nir masegosa code
echo [End]
zip -rq code.zip code
rm -r code
