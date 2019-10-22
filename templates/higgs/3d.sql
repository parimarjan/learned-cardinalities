SELECT COUNT(*)
FROM
higgs
WHERE 'Xcol22' < higgs.col22
AND higgs.col22 <= 'Ycol22'
AND 'Xcol23' < higgs.col23
AND higgs.col23 <= 'Ycol23'
AND 'Xcol27' < higgs.col27
AND higgs.col27 <= 'Ycol27'
