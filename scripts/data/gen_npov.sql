SELECT 
	rev_id, rev_comment, rev_user, rev_user_text, rev_timestamp, rev_minor_edit
FROM 
	revision INNER JOIN page 
ON 
	page_id = rev_page
WHERE
	page_namespace = 0 AND 
	rev_comment RLIKE "\\b(n?pov|neutral)\\b" AND 
	rev_comment NOT RLIKE "\\bReverted|Undid\\b"

