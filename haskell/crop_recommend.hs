{-# LANGUAGE DeriveGeneric #-}

import Data.List     (sortOn, groupBy, intercalate, nub)
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import GHC.Generics
import Text.Read     (readMaybe)
import Data.Maybe    (mapMaybe)
import Data.Char     (toLower)
import System.IO     (hSetBuffering, stdin, BufferMode(..))
import Control.Monad (when)
import Text.Printf   (printf)

-- ============================================
-- FORECAST
-- ============================================

data Forecast = Forecast
  { enso_phase     :: String
  , enso_intensity :: String
  , rainfall_input :: Double
  } deriving (Show, Generic)

instance FromJSON Forecast

loadForecast :: IO Forecast
loadForecast = do
  result <- B.readFile "../forecast.json"
  case decode result of
    Just f  -> return f
    Nothing -> return (Forecast "Neutral" "Weak" 800)

-- ============================================
-- DATA
-- ============================================

data CropRecord = CropRecord
  { crop          :: String
  , year          :: Int
  , season        :: String
  , state         :: String
  , yield         :: Double
  , rainfall      :: Double
  , avgONI        :: Double
  , ensoPhase     :: String
  , avgYield      :: Double
  , stdYield      :: Double
  , rainfallDev   :: Double
  , ensoIntensity :: String
  , yearWeight    :: Double
  , cropCount     :: Int
  } deriving (Show)

-- ============================================
-- CSV PARSER
-- ============================================

splitCsvLine :: String -> [String]
splitCsvLine [] = []
splitCsvLine s  = go s False []
  where
    go []        _    cur = [trimStr (reverse cur)]
    go ('"':xs)  True cur = go xs False cur
    go ('"':xs) False cur = go xs True  cur
    go (',':xs) False cur = trimStr (reverse cur) : go xs False []
    go (c:xs)    inQ  cur = go xs inQ (c : cur)

trimStr :: String -> String
trimStr = f . f
  where f = reverse . dropWhile (== ' ')

safeGet :: Int -> [String] -> String
safeGet i xs = if i < length xs then xs !! i else ""

readDbl :: Double -> String -> Double
readDbl d s = maybe d id (readMaybe s)

readInt :: Int -> String -> Int
readInt d s = maybe d id (readMaybe s)

toRecordMaybe :: [String] -> Maybe CropRecord
toRecordMaybe xs
  | length xs < 14 = Nothing
  | otherwise = Just $ CropRecord
      (safeGet 0 xs)
      (readInt 0 (safeGet 1  xs))
      (safeGet 2 xs)
      (safeGet 3 xs)
      (readDbl 0 (safeGet 4  xs))
      (readDbl 0 (safeGet 5  xs))
      (readDbl 0 (safeGet 6  xs))
      (safeGet 7 xs)
      (readDbl 0 (safeGet 8  xs))
      (readDbl 0 (safeGet 9  xs))
      (readDbl 0 (safeGet 10 xs))
      (safeGet 11 xs)
      (readDbl 1 (safeGet 12 xs))
      (readInt 1 (safeGet 13 xs))

loadData :: FilePath -> IO [CropRecord]
loadData file = do
  contents <- readFile file
  return $ mapMaybe (toRecordMaybe . splitCsvLine) (tail $ lines contents)

cleanData :: [CropRecord] -> [CropRecord]
cleanData = filter (\r -> yield r > 0 && rainfall r > 0)

-- ============================================
-- HELPERS
-- ============================================

normalizeStr :: String -> String
normalizeStr = map toLower . filter (/= ' ')

clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)

r2 :: Double -> String
r2 x = printf "%.2f" x

-- ============================================
-- STATISTICS
-- ============================================

safeMean :: [Double] -> Double
safeMean [] = 0
safeMean xs = sum xs / fromIntegral (length xs)

safeStd :: [Double] -> Double
safeStd xs
  | length xs < 2 = 0
  | otherwise =
      let m = safeMean xs
      in sqrt $ safeMean (map (\x -> (x - m) ** 2) xs)

-- Raw CV: not clamped so we can detect CV > 1.0
rawCV :: [Double] -> Double
rawCV xs
  | null xs          = 0
  | safeMean xs == 0 = 0
  | otherwise        = safeStd xs / abs (safeMean xs)

-- Exponential scaling: gives meaningful separation across the CV range.
--   CV=0.10 → 0.20   CV=0.25 → 0.43   CV=0.40 → 0.59
--   CV=0.60 → 0.74   CV=1.00 → 0.89
scaledCV :: Double -> Double
scaledCV cv = clamp 0 1 (1 - exp (- cv / 0.45))

safeCorr :: [Double] -> [Double] -> Double
safeCorr xs ys
  | length xs /= length ys = 0
  | length xs < 5          = 0
  | sdX == 0 || sdY == 0   = 0
  | otherwise =
      let mx  = safeMean xs
          my  = safeMean ys
          n   = fromIntegral (length xs)
          num = sum (zipWith (\x y -> (x - mx) * (y - my)) xs ys)
      in clamp (-1) 1 (num / (n * sdX * sdY))
  where
    sdX = safeStd xs
    sdY = safeStd ys

safeTrendSlope :: [CropRecord] -> Double
safeTrendSlope rs
  | length rs < 3 = 0
  | den == 0      = 0
  | otherwise     =
      let ys  = map yield rs
          ts  = map (fromIntegral . year) rs
          mt  = safeMean ts
          my  = safeMean ys
          num = sum (zipWith (\t y -> (t - mt) * (y - my)) ts ys)
      in num / den
  where
    ts  = map (fromIntegral . year) rs
    mt  = safeMean ts
    den = sum (map (\t -> (t - mt) ** 2) ts)

-- ============================================
-- FILTER
-- ============================================

filterBase :: String -> String -> [CropRecord] -> [CropRecord]
filterBase st seas =
  filter (\r ->
    normalizeStr (state  r) == normalizeStr st &&
    normalizeStr (season r) == normalizeStr seas
  )

findCropSeasons :: String -> String -> [CropRecord] -> [String]
findCropSeasons st cropName dataset =
  nub
  . map season
  . filter (\r ->
      normalizeStr (state r) == normalizeStr st &&
      normalizeStr (crop  r) == normalizeStr cropName
    )
  $ dataset

-- ============================================
-- ENSO / CROP CHARACTERISTICS
-- ============================================

elNinoSensitive :: [String]
elNinoSensitive = ["rice","sugarcane","jute","banana","coconut","tea"]

laNinaBeneficiary :: [String]
laNinaBeneficiary = ["rice","sugarcane","jute","tea"]

droughtTolerant :: [String]
droughtTolerant = ["bajra","jowar","groundnut","cotton","sorghum","millets"]

-- Crops with known price/market volatility that amplifies yield risk
highVolatilityCrops :: [String]
highVolatilityCrops = ["onion","potato","tomato","peas&beans","garlic","chilli"]

ensoEffect :: String -> String -> Double -> Double
ensoEffect phase cropName base =
  let cn = normalizeStr cropName
      elFactor
        | cn `elem` map normalizeStr elNinoSensitive = -0.20
        | cn `elem` map normalizeStr droughtTolerant =  0.05
        | otherwise                                  = -0.10
      laFactor
        | cn `elem` map normalizeStr laNinaBeneficiary = 0.12
        | otherwise                                     = 0.05
  in case phase of
       "El Nino" -> elFactor * base
       "La Nina" -> laFactor * base
       _         -> 0

-- ============================================
-- PREDICTION
-- ============================================

predictYield :: String -> String -> [CropRecord] -> Double
predictYield _    _        [] = 0
predictYield enso cropName rs =
  let base = safeMean (map yield rs)
      dev  = safeMean (map rainfallDev rs)
  in max 0 (base + 0.2 * dev + ensoEffect enso cropName base)

-- ============================================
-- RISK MODEL
-- ============================================

{-
  THE CONSISTENCY BUG (now fixed):
  ─────────────────────────────────────────────────────────────
  Previously, the alternatives pool was built using topSimilar,
  which filtered records by rainfall/ENSO similarity BEFORE
  grouping and ranking. This meant:

    Selected crop  → computeRisk uses ALL its records from baseFiltered
    Same crop as alt → computeRisk used only the topSimilar SUBSET

  Different record subsets → different CV, mean, std → different
  risk score for the SAME crop depending on which path it went through.
  This is why Onion scored 0.48 as "selected" but would show lower
  if it appeared in the alternatives list.

  THE FIX:
  ─────────────────────────────────────────────────────────────
  topSimilar is now ONLY used to decide which crop NAMES to include
  in the alternatives. Once the names are selected, ALL records for
  those crops are fetched from baseFiltered for risk computation.

  This guarantees:
    computeRisk(crop, baseFiltered records) is identical
    whether the crop is "selected" or "an alternative".

  Risk model weights (sum = 1.0):
    F1 Yield Variability  x 0.35  (scaledCV — exponential, not linear)
    F2 Rainfall Deviation x 0.20
    F3 ENSO Phase Risk    x 0.25  (Neutral=0.30, not near-zero)
    F4 Yield Trend Risk   x 0.10
    F5 Decoupling Risk    x 0.10

  Flat penalties:
    Sparsity   : n<3→+0.15, n<5→+0.10, n<10→+0.05
    Extreme CV : rawCV>=1.0 → +0.15
    Volatility : known volatile crop → +0.08

  Safe threshold: 0.40  (MODERATE crops are NOT safe alternatives)
-}

data RiskBreakdown = RiskBreakdown
  { rbRawCV      :: Double
  , rbF1         :: Double
  , rbF2         :: Double
  , rbF3         :: Double
  , rbF4         :: Double
  , rbF5         :: Double
  , rbSparsity   :: Double
  , rbHighCV     :: Double
  , rbVolatility :: Double
  , rbTotal      :: Double
  } deriving (Show)

computeRisk :: String -> String -> Double -> [CropRecord] -> RiskBreakdown
computeRisk enso cropName forecastRain rs =
  let
    n        = length rs
    ys       = map yield rs
    rainVals = map rainfall rs
    yMean    = safeMean ys
    rainMean = safeMean rainVals
    cn       = normalizeStr cropName

    cv = rawCV ys
    f1 = scaledCV cv

    f2 = clamp 0 1 (abs (forecastRain - rainMean) / (rainMean + 1))

    f3 = case enso of
           "El Nino" -> 1.00
           "La Nina" -> 0.65
           _         -> 0.30   -- Neutral still carries baseline uncertainty

    slope = safeTrendSlope rs
    f4    = if yMean <= 0 || n < 3
            then 0.20           -- unknown trend: small baseline penalty
            else clamp 0 1 (max 0 ((-slope / yMean) * 5))

    corr = safeCorr rainVals ys
    f5   = if n < 5 then 0.55
           else clamp 0 1 (1 - max 0 corr)

    sparsity
      | n < 3     = 0.15
      | n < 5     = 0.10
      | n < 10    = 0.05
      | otherwise = 0.0

    highCVPenalty     = if cv >= 1.0 then 0.15 else 0.0
    volatilityPenalty =
      if cn `elem` map normalizeStr highVolatilityCrops then 0.08 else 0.0

    weighted = 0.35 * f1
             + 0.20 * f2
             + 0.25 * f3
             + 0.10 * f4
             + 0.10 * f5

    total = weighted + sparsity + highCVPenalty + volatilityPenalty

  in RiskBreakdown
       { rbRawCV      = cv
       , rbF1         = f1
       , rbF2         = f2
       , rbF3         = f3
       , rbF4         = f4
       , rbF5         = f5
       , rbSparsity   = sparsity
       , rbHighCV     = highCVPenalty
       , rbVolatility = volatilityPenalty
       , rbTotal      = clamp 0 1 total
       }

safeThreshold :: Double
safeThreshold = 0.40

riskLabel :: Double -> String
riskLabel r
  | r <= 0.35 = "LOW"
  | r <= 0.50 = "MODERATE"
  | r <= 0.65 = "HIGH"
  | otherwise = "VERY HIGH"

plantingAdvice :: Double -> String -> String
plantingAdvice risk enso
  | risk <= 0.35 =
      "SAFE TO PLANT — low risk under current " ++ enso ++ " conditions."
  | risk <= 0.50 =
      "GENERALLY SAFE — moderate risk. Monitor rainfall and ENSO advisories."
  | risk <= 0.65 =
      "USE CAUTION — high risk under " ++ enso ++ " conditions.\n"
      ++ "               Consider alternatives or delay to a safer season."
  | otherwise =
      "NOT ADVISABLE — very high risk under " ++ enso ++ " conditions.\n"
      ++ "               Strongly consider switching crops this season."

-- ============================================
-- GROUPING & RANKING
-- ============================================

groupCrops :: [CropRecord] -> [[CropRecord]]
groupCrops = groupBy (\a b -> crop a == crop b) . sortOn crop

-- Rank every crop in the pool using ALL its records from baseFiltered.
-- baseFiltered is passed separately so records are never subset-filtered
-- by similarity — similarity only controls which crop names are in altNames.
rankCropsConsistent
  :: [String]       -- altNames: which crop names to rank
  -> [CropRecord]   -- baseFiltered: full record pool for this state+season
  -> String         -- enso phase
  -> Double         -- forecast rainfall
  -> [(String, Double, Double, RiskBreakdown)]
rankCropsConsistent altNames baseFiltered enso forecastRain =
  [ (c, rbTotal rb, predictYield enso c recs, rb)
  | c <- altNames
  , let recs = filter (\r -> normalizeStr (crop r) == normalizeStr c) baseFiltered
  , not (null recs)
  , let rb = computeRisk enso c forecastRain recs
  ]

-- ============================================
-- DISPLAY
-- ============================================

printCropLine :: (String, Double, Double, RiskBreakdown) -> IO ()
printCropLine (c, r, y, _) =
  putStrLn $ "  " ++ c
          ++ "  |  Risk: " ++ r2 r ++ " (" ++ riskLabel r ++ ")"
          ++ "  |  Est. Yield: " ++ r2 y ++ " tons/ha"

printBreakdown :: RiskBreakdown -> IO ()
printBreakdown rb = do
  putStrLn "  --- Risk Breakdown (factor x weight = contribution) ---"
  putStrLn $ "  F1 Yield Variability  raw CV=" ++ r2 (rbRawCV rb)
          ++ "  scaled=" ++ r2 (rbF1 rb)
          ++ "  x0.35 = " ++ r2 (0.35 * rbF1 rb)
  putStrLn $ "  F2 Rainfall Deviation " ++ r2 (rbF2 rb)
          ++ "                        x0.20 = " ++ r2 (0.20 * rbF2 rb)
  putStrLn $ "  F3 ENSO Phase Risk    " ++ r2 (rbF3 rb)
          ++ "                        x0.25 = " ++ r2 (0.25 * rbF3 rb)
  putStrLn $ "  F4 Yield Trend Risk   " ++ r2 (rbF4 rb)
          ++ "                        x0.10 = " ++ r2 (0.10 * rbF4 rb)
  putStrLn $ "  F5 Decoupling Risk    " ++ r2 (rbF5 rb)
          ++ "                        x0.10 = " ++ r2 (0.10 * rbF5 rb)
  when (rbSparsity   rb > 0) $
    putStrLn $ "  Sparsity Penalty      (flat)                  = "
            ++ r2 (rbSparsity rb)
  when (rbHighCV     rb > 0) $
    putStrLn $ "  High-CV Penalty       (flat, raw CV >= 1.0)   = "
            ++ r2 (rbHighCV rb)
  when (rbVolatility rb > 0) $
    putStrLn $ "  Volatility Penalty    (flat, market-volatile) = "
            ++ r2 (rbVolatility rb)
  putStrLn   "  ─────────────────────────────────────────────────────"
  putStrLn $ "  TOTAL RISK SCORE                               = "
          ++ r2 (rbTotal rb) ++ "  (" ++ riskLabel (rbTotal rb) ++ ")"
  when (rbF2 rb >= 0.99) $
    putStrLn "  *** Severe rainfall mismatch — crop water needs far exceed forecast ***"

-- ============================================
-- MAIN
-- ============================================

main :: IO ()
main = do
  hSetBuffering stdin LineBuffering

  st          <- getLine
  seasonInput <- getLine
  cr          <- getLine

  raw <- loadData "../final_crop_dataset_complete.csv"
  let dataset = cleanData raw

  forecast <- loadForecast
  let enso      = enso_phase forecast
  let rainInput = rainfall_input forecast

  putStrLn "\n=============================="
  putStrLn "  ENSO CROP RISK ADVISOR"
  putStrLn "=============================="
  putStrLn $ "State    : " ++ st
  putStrLn $ "Season   : " ++ seasonInput
  putStrLn $ "Crop     : " ++ cr
  putStrLn $ "ENSO     : " ++ enso
  putStrLn $ "Rainfall : " ++ r2 rainInput ++ " mm"
  putStrLn $ "Safe if  : risk <= " ++ r2 safeThreshold
  putStrLn "------------------------------"

  -- All records for this state+season — the single source of truth
  let baseFiltered = filterBase st seasonInput dataset

  if null baseFiltered
    then do
      putStrLn "WARNING: No records found for this State + Season combination."
      putStrLn "         Please verify your inputs."

    else do
      -- Selected crop's records — pulled directly from baseFiltered
      let selectedRecords =
            filter (\r -> normalizeStr (crop r) == normalizeStr cr) baseFiltered

      -- ==============================
      -- SELECTED CROP ASSESSMENT
      -- ==============================
      if null selectedRecords
        then do
          let otherSeasons = findCropSeasons st cr dataset
          if null otherSeasons
            then do
              putStrLn $ "STATUS: '" ++ cr ++ "' is not cultivated in "
                      ++ st ++ " at all."
              putStrLn   "        It may not be agro-climatically suitable for this state."
              putStrLn   "        Showing safer alternatives for this season below."
            else do
              putStrLn $ "STATUS: '" ++ cr ++ "' is NOT grown in " ++ st
                      ++ " during the " ++ seasonInput ++ " season."
              putStrLn $ "        It IS cultivated in " ++ st
                      ++ " during: " ++ intercalate ", " otherSeasons
              putStrLn   "        Consider switching to one of those seasons instead."
              putStrLn   "        Showing safer alternatives for this season below."

        else do
          -- computeRisk uses ALL records from baseFiltered for this crop
          let rb   = computeRisk enso cr rainInput selectedRecords
              risk = rbTotal rb
              yld  = predictYield enso cr selectedRecords
              n    = length selectedRecords

          putStrLn $ "Records used   : " ++ show n
          when (n < 10) $
            putStrLn "  (Note: fewer than 10 records — sparsity penalty applied)"
          when (yld < 0.1) $
            putStrLn "  (Note: extremely low yield crop — interpret predictions with caution)"
          putStrLn $ "Predicted Yield: " ++ r2 yld ++ " tons/ha"
          putStrLn $ "Risk Score     : " ++ r2 risk ++ " / 1.0"
          putStrLn $ "Planting Advice: " ++ plantingAdvice risk enso
          putStrLn ""
          printBreakdown rb

      putStrLn ""
      putStrLn "=============================="

      -- ==============================
      -- ALTERNATIVES — consistent record usage
      --
      -- Step 1: use topSimilar ONLY to pick which crop names to consider.
      --         This gives us the most climatically relevant crops.
      -- Step 2: rank each crop using ALL its records from baseFiltered —
      --         NOT the topSimilar subset.
      --
      -- This guarantees the same crop gets the exact same risk score
      -- whether it is the "selected" crop or appears as an alternative.
      -- ==============================

      -- Step 1: get unique crop names from the similarity-ordered pool
      -- (excluding the selected crop)
      let altNames =
            nub
            . map crop
            . filter (\r -> normalizeStr (crop r) /= normalizeStr cr)
            . take 200                            -- reasonable cap
            . reverse
            . sortOn (\r ->                       -- sort by similarity inline
                let rainScore = exp (- abs (rainfall r - rainInput) / 300)
                    ensoScore = if ensoPhase r == enso then 1.0 else 0.5
                in 0.6 * rainScore + 0.4 * ensoScore)
            $ baseFiltered

      -- Step 2: rank using ALL records per crop from baseFiltered
      let allRanked = rankCropsConsistent altNames baseFiltered enso rainInput

      -- Selected crop risk (used to define "strictly safer")
      let selectedRisk =
            if null selectedRecords then 1.0
            else rbTotal (computeRisk enso cr rainInput selectedRecords)

      -- Safer alternatives: strictly lower risk than selected AND below threshold
      let safeCrops =
            sortOn (\(_,r,_,_) -> r)
            . filter (\(_,r,_,_) -> r < selectedRisk && r <= safeThreshold)
            $ allRanked

      -- Riskiest: highest risk first
      let riskyRanked = reverse . sortOn (\(_,r,_,_) -> r) $ allRanked

      putStrLn "Top 3 Safer Alternatives (lower risk than your crop):"
      if null safeCrops
        then do
          let fallback = take 3 . sortOn (\(_,r,_,_) -> r) $ allRanked
          if null fallback
            then putStrLn "  No alternative crops available for this State + Season."
            else do
              putStrLn "  (No crop is strictly safer under these exact conditions."
              putStrLn "   Lowest-risk options available:)"
              mapM_ printCropLine fallback
        else mapM_ printCropLine (take 3 safeCrops)

      putStrLn ""
      putStrLn "Top 3 Riskiest Crops (avoid these):"
      if null allRanked
        then putStrLn "  No other crops to compare."
        else mapM_ printCropLine (take 3 riskyRanked)

      putStrLn "=============================="