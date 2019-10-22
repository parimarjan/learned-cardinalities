SELECT COUNT(*)
FROM
higgs
WHERE 'Xcol22' < higgs.col22
AND higgs.col22 <= 'Ycol22'
AND 'Xcol27' < higgs.col27
AND higgs.col27 <= 'Ycol27'
AND 'Xcol28' < higgs.col28
AND higgs.col28 <= 'Ycol28'
