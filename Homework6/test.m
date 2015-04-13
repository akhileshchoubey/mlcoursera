function [result] = test(email)

	X = ['dog' 'cat' 'cow' 'rat' 'fish'];
	input=['man' 'dog'];

	for i = input,
		for word = X,
			result = strcmp(i, word);
			if(result = 1)
				disp(i)
			end
		end
	end
%==============================	
end