
docs:
	pdoc --html  --force --output-dir docs pycelp
	mv docs/pycelp/* docs
	rmdir docs/pycelp
