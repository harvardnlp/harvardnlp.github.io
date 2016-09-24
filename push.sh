jekyll build
chmod 777 images/*
scp -r _site/* srush@pub-aws.seas.harvard.edu:/seas/web/sites_static/nlp.seas.harvard.edu/htdocs/
