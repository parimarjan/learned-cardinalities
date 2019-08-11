SELECT COUNT(*)
FROM
higgs
WHERE 'Xcol21' < higgs.col21
AND higgs.col21 <= 'Ycol21'
AND 'Xcol25' < higgs.col25
AND higgs.col25 <= 'Ycol25'
AND 'Xcol26' < higgs.col26
AND higgs.col26 <= 'Ycol26'
