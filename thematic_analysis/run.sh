#! /bin/sh

cd /app || exit

echo "Running _01_create_vector_store.py"
python _01_create_vector_store.py || exit

echo "Running _02_generate_codes.py"
python _02_generate_codes.py || exit

echo "Running _03_generate_broad_themes.py"
python _03_generate_broad_themes.py || exit

echo "Running _04_review_themes.py"
python _04_review_themes.py || exit
