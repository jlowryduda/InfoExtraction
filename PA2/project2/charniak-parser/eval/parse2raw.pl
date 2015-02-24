#!/usr/local/bin/perl

#usage: parse2raw.pl <TB files>
#Take treebank files and strip the brackets. just keep the raw data, including the traces

($#ARGV >= 0) || die("Usage: parse2raw.pl <trace|notrace><tb files>\n");

if(! -e "outputdir"){
    `mkdir outputdir`;
 }

$trace = 0;

if($ARGV[0] eq "trace"){
    $trace = 1;
}

my $start = 0;
if($trace){
    $start = 1;
}

foreach my $fileno ($start..$#ARGV){
    my $file = $ARGV[$fileno];
    my @parts = split(/\//, $file);
    my $basefilename = $parts[$#parts];
    my $thesentence = "";
    print $file, "\n";
    open(F, $file);
    my $subdir = substr($basefilename, 4, 2); 
    if(! -e "outputdir/$subdir"){
	`mkdir outputdir/$subdir`;
    }
    open(OUT, ">outputdir/$subdir/$basefilename");
    while(<F>){

	if(/^\(\s*\([^\t\r\n\f ]+ /){ #start of a sentence and the last sentence
	    if(!($thesentence =~ /^\s*$/)){ #end of a sentence
		my @words; #words in the sentence
		$thesentence =~ s/\(([^()\n\t\f\r ]+)[\n\t\r\f ]+([^()\n\t\n\f ]+)\s*\)/{
		    if(!isTrace($1) || $trace){
			push(@words, $2);
		    }
		    "{".$1." ".$2."}";
		}/eg;
		print OUT "<s> @words </s>\n";
	    }


	    $thesentence = $_;
	}elsif(!/^\s*<.*>\s*$/){ #not an sgml line
	    $thesentence .= $_;
	}
    }

    #get the last sentence
    if($thesentence ne ""){ #end of a sentence
	my @words; #words in the sentence
	$thesentence =~ s/\(([^()\n\t\f\r ]+)[\n\t\r\f ]+([^()\n\t\n\f ]+)\s*\)/{
	    if(!isTrace($1) || $trace){
		push(@words, $2);
	    }
	    "{".$1." ".$2."}";
	}/eg;
	print OUT "<s> @words </s>\n";
    }

    close(F);
    close(OUT);
}


sub isTrace{
    my ($pos) = @_;
    if($pos eq "-NONE-"){
	return 1;
    }
    return 0;
}
