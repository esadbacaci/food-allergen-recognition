-- Allergen database v2.0 -> v2.1: corrections from the blind dietitian audit (run once).
SET NAMES utf8mb4;
START TRANSACTION;
UPDATE yiyecekalerjenleri SET Miktar='Olabilir', Kaynak='expert', Surum='2.1' WHERE YiyecekID=97 AND AlerjenID=3;
UPDATE yiyecekalerjenleri SET Miktar='Olabilir', Kaynak='expert', Surum='2.1' WHERE YiyecekID=63 AND AlerjenID=1;
UPDATE yiyecekalerjenleri SET Miktar='Olabilir', Kaynak='expert', Surum='2.1' WHERE YiyecekID=128 AND AlerjenID=5;
UPDATE yiyecekalerjenleri SET Miktar='Sıklıkla', Kaynak='expert', Surum='2.1' WHERE YiyecekID=176 AND AlerjenID=2;
INSERT INTO yiyecekalerjenleri (YiyecekID, AlerjenID, Miktar, Kaynak, Surum) VALUES (201, 6, 'Sıklıkla', 'expert', '2.1');
INSERT INTO yiyecekalerjenleri (YiyecekID, AlerjenID, Miktar, Kaynak, Surum) VALUES (201, 5, 'Sıklıkla', 'expert', '2.1');
UPDATE yiyecekalerjenleri SET Miktar='Olabilir', Kaynak='expert', Surum='2.1' WHERE YiyecekID=155 AND AlerjenID=3;
UPDATE yiyecekalerjenleri SET Miktar='Sıklıkla', Kaynak='expert', Surum='2.1' WHERE YiyecekID=117 AND AlerjenID=1;
UPDATE yiyecekalerjenleri SET Miktar='Olabilir', Kaynak='expert', Surum='2.1' WHERE YiyecekID=117 AND AlerjenID=12;
UPDATE yiyecekalerjenleri SET Miktar='Sıklıkla', Kaynak='expert', Surum='2.1' WHERE YiyecekID=111 AND AlerjenID=3;
UPDATE yiyecekalerjenleri SET Miktar='Sıklıkla', Kaynak='expert', Surum='2.1' WHERE YiyecekID=165 AND AlerjenID=1;
REPLACE INTO versiyonlar VALUES ('2.1', CURDATE(), 'v2.1: corrections from the blind dietitian audit (60 dishes)');
COMMIT;
SELECT COUNT(*) AS iliski FROM yiyecekalerjenleri;  -- beklenen: 666
