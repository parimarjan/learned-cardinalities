CREATE INDEX IF NOT EXISTS dmv3 ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color);
CREATE INDEX IF NOT EXISTS dmv_scofflaw ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color,suspension_indicator);
CREATE INDEX IF NOT EXISTS dmv_suspension ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color,suspension_indicator);
CREATE INDEX IF NOT EXISTS dmv_revoc ON dmv (record_type,registration_class,state,body_type,fuel_type,reg_valid_date,color,revocation_indicator);
CREATE INDEX IF NOT EXISTS dmv_4a ON dmv (record_type,registration_class,state,body_type,fuel_type,color);
CREATE INDEX IF NOT EXISTS dmv_4b ON dmv (record_type,registration_class,body_type,fuel_type,reg_valid_date,color);
