#!/usr/bin/perl -w

my $WSJDIR = "/home/verbs/shared/corpora/wsj-old";
my $RAWDIR= "/home/verbs/guest/xuen/bin/tools/charniak-parser/TREEBANK/raw";
my $PARSEDIR= "/home/verbs/guest/xuen/bin/tools/charniak-parser/TREEBANK/parsed";
my $charniak_basedir = "/home/verbs/guest/xuen/bin/tools/charniak-parser";

my $rawdirs = `ls $RAWDIR`;
#print $rawdir;
my @rawdirs = split(/\n/, $rawdirs);
#print "@rawdirs\n";

foreach my $rawdir (@rawdirs){
    print "Processing ", $rawdir, "...\n";
    print "Collecting training data...\n";
    my $wsjdirs = `ls $WSJDIR`;
    `touch $rawdir.traindata`;
    my @wsjdirs = split(/\n/, $wsjdirs);
    foreach my $wsjdir (@wsjdirs){

	if($wsjdir ne $rawdir && $wsjdir ne "CVS" && $wsjdir ne "25" && $wsjdir ne "24"){
	    my $devdir = "01";
	    if ($rawdir eq "01"){
		$devdir = "02";
	    }
	    if($wsjdir ne $devdir){
	    	`cat $WSJDIR/$wsjdir/*.mrg >> $rawdir.traindata`;
	    }

	    `cat $WSJDIR/$devdir/*.mrg > $rawdir.devdata`;
	}
    }
    `$charniak_basedir/TRAIN/allScript En $charniak_basedir/DATA/EN/ $rawdir.traindata $rawdir.devdata`;
    `rm $rawdir.traindata`;
    my $rawfiles = `ls $RAWDIR/$rawdir`;
    #print $rawfiles;
    my @rawfiles = split(/\n/, $rawfiles);
    if (! -e "$PARSEDIR/$rawdir"){
	`mkdir $PARSEDIR/$rawdir`;
    }
    
    foreach my $rawfile (@rawfiles){
	print "Parse $rawfile...\n";
	`$charniak_basedir/PARSE/parseIt $charniak_basedir/DATA/EN/ $RAWDIR/$rawdir/$rawfile > $PARSEDIR/$rawdir/$rawfile.out`;

    }
    
}