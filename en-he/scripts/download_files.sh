#!/bin/bash
# Downloads WMT17 training and test data for EN-DE
# Distributed under MIT license

script_dir=`dirname $0`
script_dir=/cs/usr/bareluz/gabi_labs/nematus_clean/nematus/en-he/scripts/

main_dir=$script_dir/..

# variables (toolkits; source and target language)
. $main_dir/vars

# get EN-DE training data for WMT17

if [ ! -f $main_dir/downloads/de-en.tgz ];
then
  wget http://www.statmt.org/europarl/v7/de-en.tgz -O $main_dir/downloads/de-en.tgz
  tar -xf $main_dir/downloads/de-en.tgz -C $main_dir/downloads
fi

if [ ! -f $main_dir/downloads/training-parallel-commoncrawl.tgz ];
then
  wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz -O $main_dir/downloads/training-parallel-commoncrawl.tgz
  tar -xf $main_dir/downloads/training-parallel-commoncrawl.tgz -C $main_dir/downloads
fi

if [ ! -f $main_dir/downloads/training-parallel-nc-v12.tgz ];
then
  wget http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz -O $main_dir/downloads/training-parallel-nc-v12.tgz
  tar -xf $main_dir/downloads/training-parallel-nc-v12.tgz -C $main_dir/downloads
fi

if [ ! -f $main_dir/downloads/rapid2016.tgz ];
then
  wget http://data.statmt.org/wmt17/translation-task/rapid2016.tgz -O $main_dir/downloads/rapid2016.tgz
  tar -xf $main_dir/downloads/rapid2016.tgz -C $main_dir/downloads
fi

if [ ! -f $main_dir/downloads/dev.tgz ];
then
  wget http://data.statmt.org/wmt17/translation-task/dev.tgz -O $main_dir/downloads/dev.tgz
  tar -xf $main_dir/downloads/dev.tgz -C $main_dir/downloads
fi

if [ ! -f $main_dir/downloads/test.tgz ];
then
  wget http://data.statmt.org/wmt17/translation-task/test.tgz -O $main_dir/downloads/test.tgz
  tar -xf $main_dir/downloads/test.tgz -C $main_dir/downloads
fi


# concatenate all training corpora
cat $main_dir/downloads/europarl-v7.de-en.en $main_dir/downloads/commoncrawl.de-en.en $main_dir/downloads/rapid2016.de-en.en $main_dir/downloads/training/news-commentary-v12.de-en.en > $main_dir/data/corpus.en
cat $main_dir/downloads/europarl-v7.de-en.de $main_dir/downloads/commoncrawl.de-en.de $main_dir/downloads/rapid2016.de-en.de $main_dir/downloads/training/news-commentary-v12.de-en.de > $main_dir/data/corpus.de

for year in 2013;
do
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newstest${year}-ref.de.sgm > $main_dir/data/newstest$year.de
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newstest${year}-src.en.sgm > $main_dir/data/newstest$year.en
done

for year in 2014;
do
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newstest${year}-deen-ref.de.sgm > $main_dir/data/newstest$year.de
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newstest${year}-deen-src.en.sgm > $main_dir/data/newstest$year.en
done

for year in {2015,2016};
do
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newstest${year}-ende-ref.de.sgm > $main_dir/data/newstest$year.de
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/dev/newstest${year}-ende-src.en.sgm > $main_dir/data/newstest$year.en
done

for year in 2017;
do
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/test/newstest${year}-ende-ref.de.sgm > $main_dir/data/newstest$year.de
  $moses_scripts/ems/support/input-from-sgm.perl < $main_dir/downloads/test/newstest${year}-ende-src.en.sgm > $main_dir/data/newstest$year.en
done


cd ..
