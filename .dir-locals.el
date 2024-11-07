((julia-mode . ((julia-snail-port . 10050)
                (julia-snail-repl-buffer . "*julia Venus*")
                (julia-snail-executable . "nix develop --impure --command julia")))

(org-mode . ((julia-snail-port . 10060)
                (julia-snail-repl-buffer . "*julia*")
                (julia-snail-executable . "nix develop --impure --command julia"))))
