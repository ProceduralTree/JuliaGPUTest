((julia-mode . ((julia-snail-port . 10050)
                (julia-snail-repl-buffer . "*julia Venus*")
                (julia-snail-executable . "julia")))

(org-mode . ((julia-snail-port . 10060)
                (julia-snail-repl-buffer . "*julia Org*")
                (julia-snail-executable . "nix develop --impure --command julia"))))
