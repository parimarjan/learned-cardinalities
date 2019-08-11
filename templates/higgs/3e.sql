SELECT COUNT(*)
FROM
higgs
WHERE 'Xcol21' < higgs.col21
AND higgs.col21 <= 'Ycol21'
AND 'Xcol22' < higgs.col22
AND higgs.col22 <= 'Ycol22'
AND 'Xcol27' < higgs.col27
AND higgs.col27 <= 'Ycol27'
